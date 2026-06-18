import aiohttp
import asyncio
from core.state import AgentState
from langchain_core.messages import HumanMessage

MEMOS_BASE_URL = "http://127.0.0.1:8000/product"

async def search_memories_node(state: AgentState) -> dict:
    """异步记忆检索节点，不阻塞事件循环"""
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
        "dedup": "mmr",
        "top_k": 5,
    }
    
    try:
        # 使用aiohttp进行异步请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MEMOS_BASE_URL}/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3, connect=1)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                memories = []
                for bucket in data.get("data", {}).get("text_mem", []):
                    for mem in bucket.get("memories", []):
                        memories.append(mem.get("memory", ""))
                
                context = "\n".join(memories) if memories else ""
                
    except asyncio.TimeoutError:
        print("[HTTP检索] 请求超时，返回空上下文")
        context = ""
    except Exception as e:
        print(f"[HTTP检索] 出错：{e}")
        context = ""
    
    return {"search_context": context}
