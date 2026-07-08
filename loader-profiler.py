import fast_langgraph
#fast_langgraph.shim.patch_langgraph()
import asyncio
import time
from core.graph import graph
from langchain_core.messages import HumanMessage, AIMessageChunk
from core.nodes.add_memories_backend import flush_memories_task
async def main():
    print("机器人已启动，输入 quit 退出")
    config = {"configurable": {"thread_id": "1"}}
    asyncio.create_task(flush_memories_task())
    loop = asyncio.get_running_loop()

    while True:
        user = await loop.run_in_executor(None, input, "\n你: ")
        if user.lower() == "quit":
            break

        print("AI: ", end="", flush=True)

        # ---------- 性能统计变量 ----------
        timings = {}               # 记录每个节点的累计耗时
        start_total = time.time()  # 本轮总开始时间
        current_node = None        # 当前处理的节点名
        node_start = None          # 当前节点开始的时间戳
        first_chunk_time = None    # 首个内容块的时间（相对于start_total）
        # ---------------------------------

        async for msg_chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user)]},
            config,
            stream_mode="messages",
        ):
            # 1. 提取节点名称（兼容dict或对象形式）
            if isinstance(metadata, dict):
                node_name = metadata.get("langgraph_node", "unknown")
            else:
                node_name = getattr(metadata, "langgraph_node", "unknown")

            # 2. 节点切换时，累计上一个节点的耗时
            if node_name != current_node:
                if current_node is not None and node_start is not None:
                    elapsed = time.time() - node_start
                    timings[current_node] = timings.get(current_node, 0) + elapsed
                current_node = node_name
                node_start = time.time()

            # 3. 处理实际的输出内容
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                content = msg_chunk.content
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_total
                print(content, end="", flush=True)

        # ---------- 记录最后一个节点的耗时 ----------
        if current_node is not None and node_start is not None:
            elapsed = time.time() - node_start
            timings[current_node] = timings.get(current_node, 0) + elapsed

        total_time = time.time() - start_total
        print()  # 换行

        # ---------- 打印性能报告 ----------
        print(f"⏱️  性能报告 (本轮对话)")
        print(f"总耗时:         {total_time * 1000:8.2f} ms")
        if first_chunk_time is not None:
            print(f"首字延迟(TTFT): {first_chunk_time * 1000:8.2f} ms")
        else:
            print("首字延迟(TTFT): 无输出内容")

        if timings:
            print(f"\n节点耗时明细:")
            for node, t in sorted(timings.items(), key=lambda x: -x[1]):
                pct = (t / total_time) * 100 if total_time > 0 else 0
                bar = "█" * int(pct / 5)
                print(f"  {node:20s} {t * 1000:8.2f} ms  ({pct:5.1f}%) {bar}")
        else:
            print("\n未捕获到任何节点耗时数据")

        print(f"{'=' * 55}\n")

if __name__ == "__main__":
    asyncio.run(main())