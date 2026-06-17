import requests
from core.state import ChatState
from langchain_core.messages import HumanMessage

# 配置 MemOS 服务地址（根据实际情况修改）
MEMOS_BASE_URL = "http://127.0.0.1:8000/product"

def search_memories_node(state: ChatState) -> dict:
    user_id = "b32d0977-435d-4828-a86f-4f47f8b55bca"
    messages = state.get("messages", [])
    
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    
    if not last_user_msg:
        return {"search_context": ""}
    
    payload = {
        "query": last_user_msg,
        "user_id": user_id,
        "readable_cube_ids": [user_id],
        "dedup":"mmr",
        "top_k": 5,
    }
    
    try:
        response = requests.post(
            f"{MEMOS_BASE_URL}/search",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        memories = []
        for bucket in data.get("data", {}).get("text_mem", []):
            for mem in bucket.get("memories", []):
                memories.append(mem.get("memory", ""))
        
        context = "\n".join(memories) if memories else ""
    except Exception as e:
        print(f"[HTTP检索] 出错：{e}")   # 保留错误日志，方便排查
        context = ""
    
    return {"search_context": context}