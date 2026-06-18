import os
import sys
import re
import queue
import threading
import subprocess
import tempfile
import asyncio
import edge_tts
from concurrent.futures import ThreadPoolExecutor
from core.graph import graph
from langchain_core.messages import HumanMessage, AIMessageChunk

# ---------- TTS 配置 ----------
TTS_VOICE = "zh-CN-XiaoyiNeural"
ENABLE_AUTO_PLAY = True
MAX_CONCURRENT_SYNTHESIS = 3   # 并行合成线程数

# ---------- 全局队列和状态 ----------
synthesis_queue = queue.Queue()          # (seq, sentence)
completed_results = {}                   # seq -> filepath
next_play_index = 0
playback_condition = threading.Condition()
stop_flag = False
executor = None

# ---------- 句子分割 ----------
def split_sentences(text: str) -> list[str]:
    sentences = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？；!?;,，、:：……":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences

# ---------- 合成函数（在子线程中运行） ----------
def synthesize_sync(seq: int, sentence: str):
    """同步合成单个句子（内部使用 asyncio.run）并保存为临时 MP3，返回路径"""
    if not sentence:
        return
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_file = f.name
    try:
        communicate = edge_tts.Communicate(sentence, TTS_VOICE,rate="+15%",pitch="+10Hz")
        asyncio.run(communicate.save(output_file))
        # 合成完成，将结果存入缓存并通知播放线程
        with playback_condition:
            completed_results[seq] = output_file
            playback_condition.notify()
    except Exception as e:
        print(f"\n⚠️ 句子 {seq} 合成失败: {e}")
        # 合成失败也要通知，否则播放线程会永远等待
        with playback_condition:
            completed_results[seq] = None   # 标记为失败，跳过
            playback_condition.notify()

# ---------- 播放线程（顺序播放） ----------
def playback_worker():
    global next_play_index
    while not stop_flag:
        with playback_condition:
            # 等待下一个序号的合成结果出现
            while next_play_index not in completed_results and not stop_flag:
                playback_condition.wait()
            if stop_flag:
                break
            filepath = completed_results.pop(next_play_index)
            next_play_index += 1
        if filepath is not None and ENABLE_AUTO_PLAY:
            try:
                # 同步播放
                if sys.platform == "win32":
                    subprocess.run(["start", "/wait", filepath], shell=True)
                elif sys.platform == "darwin":
                    subprocess.run(["afplay", filepath])
                else:
                    subprocess.run(["mpg123", "-q", filepath])
                # 播放后删除临时文件
                try:
                    os.unlink(filepath)
                except:
                    pass
            except Exception as e:
                print(f"\n⚠️ 播放失败: {e}")

# ---------- 合成线程池执行器 ----------
def synthesis_worker():
    """从合成队列取任务提交给线程池"""
    global executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SYNTHESIS)
    while not stop_flag:
        try:
            seq, sentence = synthesis_queue.get(timeout=0.5)
            if seq is None:   # 退出信号
                break
            executor.submit(synthesize_sync, seq, sentence)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n⚠️ 合成任务提交失败: {e}")

# ---------- 启动和停止 ----------
def start_tts_system():
    global stop_flag
    stop_flag = False
    # 启动播放线程
    threading.Thread(target=playback_worker, daemon=True).start()
    # 启动合成调度线程（负责从队列取任务提交给线程池）
    threading.Thread(target=synthesis_worker, daemon=True).start()

def stop_tts_system():
    global stop_flag, executor
    stop_flag = True
    # 通知合成队列退出
    synthesis_queue.put((None, None))
    if executor:
        executor.shutdown(wait=True, cancel_futures=False)
    # 唤醒播放线程
    with playback_condition:
        playback_condition.notify_all()

# ---------- 主程序 ----------
def main():
    print("🤖 机器人已启动，输入 'quit' 退出")
    print("🔊 TTS 已启用（异步合成，顺序播放）")
    config = {"configurable": {"thread_id": "1"}}
    
    start_tts_system()
    seq_counter = 0

    try:
        while True:
            user = input("\n你: ")
            if user.lower() == "quit":
                break

            print("AI: ", end="", flush=True)
            buffer = ""
            for msg_chunk, _ in graph.stream(
                {"messages": [HumanMessage(content=user)]},
                config,
                stream_mode="messages",
            ):
                if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                    print(msg_chunk.content, end="", flush=True)
                    buffer += msg_chunk.content
                    sentences = split_sentences(buffer)
                    if len(sentences) > 1:
                        for sent in sentences[:-1]:
                            if sent.strip():
                                synthesis_queue.put((seq_counter, sent))
                                seq_counter += 1
                        buffer = sentences[-1]
            print()
            # 处理最后剩余的 buffer
            if buffer.strip():
                synthesis_queue.put((seq_counter, buffer.strip()))
                seq_counter += 1

            # 等待所有句子合成完成（可选项）
            # 这里不等待，允许后台继续合成播放，用户可立即输入下一轮
    finally:
        stop_tts_system()
        print("👋 已退出")

if __name__ == "__main__":
    main()