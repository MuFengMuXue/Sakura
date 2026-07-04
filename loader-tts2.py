# ========== 1. 全局 Rust 加速 ==========
import fast_langgraph
fast_langgraph.shim.patch_langgraph()

# ========== 2. 导入 ==========
import asyncio
from core.graph import graph
from langchain_core.messages import HumanMessage, AIMessageChunk
from backend.tts.tts_loader import StreamingTTS

# ---------- 3. TTS 单例 ----------
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
    print("语音助手已启动，输入 quit 退出")

    # 启动 TTS 后台长连接（你在原代码里加了这步，非常对）
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

        # ---------- 流式接收 LLM 输出 ----------
        async for msg_chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user)]},
            config,
            stream_mode="messages",
        ):
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                content = msg_chunk.content
                print(content, end="", flush=True)

                buffer += content
                sentences, buffer = split_sentences(buffer)

                for sentence in sentences:
                    if sentence.strip():
                        # await add_sentence，不卡主循环
                        await tts.add_sentence(sentence.strip())

        print()  # 换行

        # 处理最后的不完整句子
        if buffer.strip():
            #换成 await add_sentence
            await tts.add_sentence(buffer.strip())



if __name__ == "__main__":
    asyncio.run(main())