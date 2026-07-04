import aiohttp
import asyncio
from core.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

MEMOS_BASE_URL = "http://127.0.0.1:8004"


async def add_memories_node(state: AgentState) -> dict:
    """
    将最后一轮对话写入记忆
    """
    user_id = "01"
    messages = state.get("messages",[])

    if len(messages)<2:
        return {}
    
    last_msg = messages[-1]
    prev_msg = messages[-2]

    if isinstance(last_msg, AIMessage) and isinstance(prev_msg, HumanMessage):
        user_msg = prev_msg.content
        ai_msg = last_msg.content
    else:
        return {}
    if user_msg is None or ai_msg is None:
        return {}

    user_msg = str(user_msg)
    ai_msg = str(ai_msg)

    
    # 缓存更新
    buffer = list(state.get("memory_buffer", []))
    buffer.append((user_msg,ai_msg))
    new_count = len(buffer)

    if new_count >= 5:
        messages_to_save = []
        for user , ai in buffer:
            messages_to_save.append({"role":"user","content":user})
            messages_to_save.append({"role":"assistant","content":ai})
        payload = {"messages":messages_to_save,"user_id":user_id}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{MEMOS_BASE_URL}/add",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10, connect=1)
                ) as resp:
                    if resp.status == 200:
                        print("保存成功")
                    else:
                        error_text = await resp.text()
                        print(f"保存失败，状态码: {resp.status}, 错误信息: {error_text}")
        except Exception as e:
            print(f"保存失败: {e}")
        
        return{
            "memory_buffer": [],
            "buffer_count": 0
        }
    else:
        return{
            "memory_buffer": buffer,
            "buffer_count": new_count
        }
    

