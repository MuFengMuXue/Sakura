import json
import aiofiles
from core.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from core.nodes.add_memory_manager import notification_queue, MEMOS_FILE, logger

async def add_memories_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if len(messages) < 2:
        return {}

    last_msg = messages[-1]
    prev_msg = messages[-2]

    if not (isinstance(last_msg, AIMessage) and isinstance(prev_msg, HumanMessage)):
        return {}

    user_content = str(prev_msg.content) if prev_msg.content is not None else ""
    ai_content = str(last_msg.content) if last_msg.content is not None else ""
    if not user_content or not ai_content:
        return {}

    record = {"user": user_content, "assistant": ai_content}

    try:
        async with aiofiles.open(MEMOS_FILE, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")
            await f.flush()
        logger.info("已追加一轮对话")

        # 通知后台任务文件有变动
        await notification_queue.put(None)

    except Exception as e:
        logger.error(f"写入文件失败: {e}")

    return {}