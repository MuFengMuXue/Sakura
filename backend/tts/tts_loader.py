import asyncio
import json
import numpy as np
from collections import deque
import threading
import sounddevice as sd
import websockets  # 确保安装了: pip install websockets

class StreamingTTS:
    def __init__(self, server_url="ws://localhost:9000/ws/tts", sample_rate=48000):
        self.server_url = server_url
        self.sample_rate = sample_rate
        
        # --- 声卡缓冲区状态 (纯线程安全操作) ---
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stream = None
        self.stream_started = False

        # --- 异步通信队列 ---
        self.sentence_queue = asyncio.Queue()
        self._worker_task = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """sounddevice 底层回调，原封不动保留"""
        needed = frames
        with self.buffer_lock:
            collected = []
            remaining = needed
            while remaining > 0 and self.audio_buffer:
                chunk = self.audio_buffer.popleft()
                if len(chunk) >= remaining:
                    collected.append(chunk[:remaining])
                    if len(chunk) > remaining:
                        self.audio_buffer.appendleft(chunk[remaining:])
                    remaining = 0
                else:
                    collected.append(chunk)
                    remaining -= len(chunk)
            if collected:
                data = np.concatenate(collected)
                if len(data) < needed:
                    data = np.pad(data, (0, needed - len(data)), 'constant')
                outdata[:] = data.reshape(-1, 1)
            else:
                outdata.fill(0)

    def _start_stream(self):
        """启动声卡"""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),
        )
        self.stream.start()
        self.stream_started = True

    async def _ws_worker(self):
        """
        核心后台任务：维持单个 WebSocket 长连接。
        不断从队列拿句子 -> 发给服务端 -> 流式收音频 -> 喂声卡。
        """
        try:
            async with websockets.connect(self.server_url) as ws:
                print("🔌 TTS 音频流长连接已建立")
                while not self.stop_event.is_set():
                    # 1. 阻塞等待主循环扔过来的句子
                    sentence = await self.sentence_queue.get()
                    if sentence is None:  # 收到毒药丸，准备退出
                        break
                    
                    # 2. 通过已有的长连接发给服务端（无握手延迟）
                    await ws.send(json.dumps({"text": sentence}))
                    
                    # 3. 流式接收服务端返回的音频块
                    stream_just_started = False
                    while True:
                        msg = await ws.recv()
                        if msg == b"__END__":
                            break
                        if isinstance(msg, str):  # 忽略服务端的 JSON 状态包
                            continue
                            
                        # 4. 转 numpy 并喂给本地声卡缓冲区
                        chunk = np.frombuffer(msg, dtype=np.float32)
                        with self.buffer_lock:
                            self.audio_buffer.append(chunk)
                            
                        # 首块到达时启动声卡
                        if not stream_just_started:
                            if not self.stream_started:
                                self._start_stream()
                            stream_just_started = True
                            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"TTS 网络错误: {e}")

    async def start(self):
        """在 main 里最先调用，启动后台网络监听任务"""
        self._worker_task = asyncio.create_task(self._ws_worker())

    async def add_sentence(self, text):
        """主循环切分出句子后，调用此方法扔进队列（极快，非阻塞）"""
        if text and text.strip():
            await self.sentence_queue.put(text.strip())

    async def wait_finish(self):
        """等待队列清空，且声卡缓冲区播放完毕"""
        while not self.sentence_queue.empty():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1) # 稍微留点时间让最后一点网络包处理完
        
        while not self.stop_event.is_set():
            with self.buffer_lock:
                if not self.audio_buffer:
                    break
            await asyncio.sleep(0.05)

    def stop(self):
        """彻底停止并清理"""
        self.stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.buffer_lock:
            self.audio_buffer.clear()
        self.stream_started = False
        self.stop_event.clear()