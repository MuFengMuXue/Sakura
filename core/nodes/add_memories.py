import json
from core.state import ChatState
from langchain_core.messages import HumanMessage, AIMessage
from memos.api.routers.server_router import add_memories
from memos.api.product_models import APIADDRequest


def add_memories_node(state: ChatState) -> dict:
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
    add_req = APIADDRequest(
            user_id = user_id,
            writable_cube_ids = [user_id],
            messages = [
                {"role":"user","content":user_msg},
                {"role":"assistant","content":ai_msg}
            ],
            async_mode = "async",
            mode="fine"
        )
    add_rsp = add_memories(add_req)
    print(f"写入状态{add_rsp}")
    return {}

