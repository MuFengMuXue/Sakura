from core.state import ChatState
from langchain_core.messages import HumanMessage
from memos.api.product_models import APISearchRequest
from memos_client import get_search_handler

def search_memories_node(state: ChatState) -> dict:
    """检索与最后一条用户消息相关的记忆，并存入状态"""
    user_id = "creator"
    messages = state.get("messages", [])

    # 获取最后一条用户消息
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"search_context": ""}

    # 构造搜索请求
    req = APISearchRequest(
        query=last_user_msg,
        user_id=user_id,
        top_k=5,
        dedup="mmr",
        relativity=0.5,
        readable_cube_ids=[user_id]
    )

    try:
        search_handler = get_search_handler()
        response = search_handler.handle_search_memories(req)
        # 提取记忆文本
        memories = []
        for bucket in response.data.get("text_mem", []):
            for mem in bucket.get("memories", []):
                memories.append(mem.get("memory", ""))
        
        if memories:
            context = "\n".join(memories) if memories else "", "default_user"
        else:
            context = ""
    except Exception as e:
        print(f"[检索节点] 出错：{e}")
        context = ""

    return {"search_context": context}