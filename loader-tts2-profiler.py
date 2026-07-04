# ========== 1. 必须放第一行：全局 Rust 加速 ==========
import fast_langgraph
#fast_langgraph.shim.patch_langgraph()

# ========== 2. 导入 ==========
import asyncio
import time
from core.graph import graph
from langchain_core.messages import HumanMessage, AIMessageChunk
from backend.tts.tts_loader import StreamingTTS

# ---------- 3. TTS 单例实例化（此时还未连接网络） ----------
tts = StreamingTTS(server_url="ws://localhost:9000/ws/tts")

# ---------- 4. 句子切分器 ----------
def split_sentences(text):
    sentences = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？；!?;，,~":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    return sentences, buf

# ---------- 5. 主循环 ----------
async def main():
    print("🤖 语音助手已启动，输入 quit 退出")
    
    # 启动 TTS 后台长连接任务
    await tts.start()
    
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user = input("\n你: ")
        if user.lower() == "quit":
            tts.stop()
            break

        print("AI: ", end="", flush=True)

        buffer = ""
        full_reply = ""

        # ===== 计时变量 =====
        timings = {}
        start_total = time.time()
        current_node = None
        node_start = None
        first_chunk_time = None
        # ====================

        # ---------- 流式接收 LLM 输出 ----------
        async for msg_chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user)]},
            config,
            stream_mode="messages",
        ):
            if isinstance(metadata, dict):
                node_name = metadata.get("langgraph_node", "unknown")
            else:
                node_name = getattr(metadata, "langgraph_node", "unknown")

            if node_name != current_node:
                if current_node and node_start:
                    elapsed = time.time() - node_start
                    timings[current_node] = timings.get(current_node, 0) + elapsed
                current_node = node_name
                node_start = time.time()

            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                content = msg_chunk.content

                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_total

                print(content, end="", flush=True)

                buffer += content
                full_reply += content

                sentences, buffer = split_sentences(buffer)

                for sentence in sentences:
                    if sentence.strip():
                        # 核心改动：异步扔进队列，不阻塞 LLM 流式接收
                        await tts.add_sentence(sentence)

        # ===== 记录最后一个节点 =====
        if current_node and node_start:
            elapsed = time.time() - node_start
            timings[current_node] = timings.get(current_node, 0) + elapsed

        # 处理最后的不完整句子
        if buffer.strip():
            await tts.add_sentence(buffer.strip())

        # 🚀 核心改动：等待所有句子都合成并且声卡播完，再打印报告
        await tts.wait_finish()

        total_time = time.time() - start_total
        # ============================

        print()  # 换行

        # ===== 打印性能报告 =====
        print(f"⏱️  性能报告 (本轮对话)")
        print(f"总耗时:         {total_time * 1000:8.2f} ms")
        if first_chunk_time is not None:
            print(f"首字延迟(TTFT): {first_chunk_time * 1000:8.2f} ms")
        print(f"\n节点耗时明细:")
        for node, t in sorted(timings.items(), key=lambda x: -x[1]):
            pct = (t / total_time) * 100 if total_time > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {node:20s} {t * 1000:8.2f} ms  ({pct:5.1f}%) {bar}")

        # 获取真实的队列积压情况
        queue_size = tts.sentence_queue.qsize()
        print(f"\nTTS 队列剩余:   {queue_size} 句待合成")
        print(f"{'=' * 55}\n")


if __name__ == "__main__":
    asyncio.run(main())