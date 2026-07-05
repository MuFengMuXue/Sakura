import json
import asyncio
import contextlib
import threading
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from voxcpm import VoxCPM
import torch
torch.set_float32_matmul_precision('high')

# 1. 生命周期管理 (解耦模型加载)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("TTS 服务已启动，正在后台加载模型...")
    app.state.tts_model = await asyncio.to_thread(
        VoxCPM,
        voxcpm_model_path="VoxCPM2",
        enable_denoiser=False,
        device="cuda",
    )
    print("TTS 模型加载完成，服务完全就绪")
    yield

app = FastAPI(lifespan=lifespan)



# 2. 核心桥接：同步生成器 -> 异步队列
def _sync_stream_worker(model, text, params, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: threading.Event):
    """
    这个函数在独立的线程池中运行。
    它的任务是：消费同步生成器 -> 转成字节 -> 安全地丢进异步队列。
    """
    try:
        if hasattr(model, 'reset_kv_cache'):
            model.reset_kv_cache()

        gen = model.generate_streaming(
            text=text,
            cfg_value=params.get("cfg_value", 2.0),
            inference_timesteps=params.get("inference_step", 7),
            prompt_wav_path="03.wav",
            prompt_text="这是什么鸟，好优雅呀。黑水鸡吗？",
            retry_badcase=False,
        )
        
        # 每生成一块，立刻推入队列
        for chunk in gen:
            # 检查客户端是否已经断开，如果是，立刻停止生成，节省 GPU 算力
            if stop_event.is_set():
                break
                
            audio_bytes = chunk.astype(np.float32).tobytes()
            
            # 关键：从同步线程安全地向异步队列塞数据
            asyncio.run_coroutine_threadsafe(queue.put(audio_bytes), loop)
            
    except Exception as e:
        print(f"TTS 生成线程发生错误: {e}")
    finally:
        # 无论成功还是报错，最后一定要发一个 None 作为结束信号
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)



# 3. WebSocket 接口 (纯异步，极速响应)
@app.websocket("/ws/tts")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            if not data:
                continue
            try:
                params = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "非法 JSON"}))
                continue

            text = params.get("text", "")
            if not text.strip():
                continue

            if not hasattr(app.state, "tts_model"):
                await ws.send_text(json.dumps({"status": "loading"}))
                continue

            # 准备桥梁组件
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            stop_event = threading.Event() # 用于在客户端断开时，通知生成线程立刻停止

            # 启动后台生成线程 (非阻塞，立刻返回)
            worker_task = asyncio.create_task(
                asyncio.to_thread(_sync_stream_worker, app.state.tts_model, text, params, queue, loop, stop_event)
            )


            # 真正的流式发送循环
            while True:
                # 阻塞等待队列里的数据（完全不卡事件循环，不占 GIL）
                audio_bytes = await queue.get()
                
                # 收到 None 说明生成完毕
                if audio_bytes is None:
                    break
                
                # 立刻发给前端播放
                await ws.send_bytes(audio_bytes)

            # 清理后台任务
            await worker_task
            
            # 发送结束标记
            await ws.send_bytes(b"__END__")

    except WebSocketDisconnect:
        print("客户端断开了 WebSocket 连接")
        stop_event.set() # 通知生成线程赶紧停
    except Exception as e:
        print(f"发生错误: {e}")
        stop_event.set()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        ws_ping_interval=None,
        ws_ping_timeout=None,
        timeout_keep_alive=120,
    )