import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, AIMessageChunk
from core.state import AgentState
from core.tools import memos_add_memory,memos_search_memory,memos_correct_memory
from core.load_config import get_config ,get_persona

tools = [memos_add_memory, memos_search_memory,memos_correct_memory]

def _load_llm():
    config = get_config()
    return ChatOpenAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        streaming=True,
        max_retries=3,
    )


llm = _load_llm().bind_tools(tools)
persona = get_persona()

async def chat_node(state: AgentState):
    context = state.get("search_context", "")
    
    final_system_prompt = persona
    if context:
        final_system_prompt += f"""
以下是关于沐风的历史记忆和偏好。在回答时，如果这些信息与问题相关，请优先采用它们来回答。

记忆内容：
{context}

请自然地融入这些信息，不要提及"根据记忆"等词语。
"""
 
    
    messages = [SystemMessage(content=final_system_prompt)] + state["messages"]
    
    # ---------- 核心修改：使用累加器 ----------
    full_message = None
    async for chunk in llm.astream(messages):
        if full_message is None:
            full_message = chunk
        else:
            # 累加 chunk：自动合并 content 和 tool_calls
            full_message = full_message + chunk
    
    # 如果没有任何输出（极少情况），返回空消息
    if full_message is None:
        return {"messages": [AIMessage(content="")]}
    
    
    return {"messages": [full_message]}