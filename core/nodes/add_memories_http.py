import requests
from core.state import ChatState
from langchain_core.messages import HumanMessage, AIMessage

MEMOS_BASE_URL = "http://127.0.0.1:8000/product"

def add_memories_node(state: ChatState) -> dict:
    """
    将最后一轮对话写入记忆
    """
    messages = state.get("messages",[])

    if len(messages)<2:
        return {}
    
    last_msg = messages[-1]
    prev_msg = messages[-2]

    if isinstance(last_msg,AIMessage) and isinstance(prev_msg,HumanMessage):
        ai_msg = last_msg
        user_msg = prev_msg
    else:
        return {}
    
    if not user_msg or not ai_msg:
        return {}

    user_msg = str(prev_msg.content) if prev_msg.content is not None else ""
    ai_msg = str(last_msg.content) if last_msg.content is not None else ""
    

    user_id = "b32d0977-435d-4828-a86f-4f47f8b55bca"
    payload = {
        "messages":[
            {"role":"user","content":user_msg},
            {"role":"assistant","content":ai_msg}
        ],
        "user_id":user_id,
        "writable_cube_id":[user_id],
        "async_mode":"async"
    }

    try:
        response = requests.post(
            f"{MEMOS_BASE_URL}/add",
            json=payload,
            timeout=5
        )
        if response.status_code == 200:
            print("写入成功")
        else:
            print(f"写入失败:{response.status_code}")
    except Exception as e:
        print(f"出错:{e}")
    return {}
        