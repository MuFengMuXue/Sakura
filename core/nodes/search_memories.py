import json
from memos.api.routers.server_router import search_memories
from memos.api.product_models import APISearchRequest
from core.state import AgentState
from langchain_core.messages import HumanMessage

def search_memories_node(state: AgentState) -> dict:
    
    user_id = "b32d0977-435d-4828-a86f-4f47f8b55bca"
    messages = state.get("messages",[])
    if not messages:
        return {"search_context": ""}
    
    last_msg = messages[-1]
    
    if not isinstance(last_msg,HumanMessage):
        return {"search_context": ""}
    
    last_user_msg = last_msg.content
    
    if not last_user_msg:
        return {"search_context": ""}
    
    search_req = APISearchRequest(
        user_id=user_id,
        readable_cube_ids=[user_id],
        query= last_user_msg,
        include_preference=True,
        dedup="mmr",
        top_k= 5,
    )
    search_rsp = search_memories(search_req).data
    
    memories = []
    for bucket in search_rsp.get("text_mem", []):
        for mem in bucket.get("memories", []):
            memories.append(mem.get("memory", ""))
    
    context = "\n".join(memories) if memories else ""

    return {"search_context": context}
