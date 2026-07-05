import asyncio
import fast_langgraph
fast_langgraph.shim.patch_langgraph()
from core.graph import graph
from langchain_core.messages import HumanMessage, AIMessageChunk

async def main():
    print("机器人已启动，输入 quit 退出")
    config = {"configurable": {"thread_id": "1"}}
    loop = asyncio.get_running_loop()
    # 使用 asyncio 实现非阻塞的输入读取（这里简化，用 run_in_executor 实现同步 input）
    while True:
        user = await loop.run_in_executor(None, input, "\n你: ")
        if user.lower() == "quit":
            break
        print("AI: ", end="", flush=True)
        # 改为异步流式调用 astream
        async for msg_chunk, _ in graph.astream(
            {"messages": [HumanMessage(content=user)]},
            config,
            stream_mode="messages",
        ):
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                print(msg_chunk.content, end="", flush=True)
        print()  # 换行

if __name__ == "__main__":
    asyncio.run(main())