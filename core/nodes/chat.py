import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from core.state import AgentState

load_dotenv()

def _load_llm():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        llm_cfg = config["llm"]
    return ChatOpenAI(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        base_url=llm_cfg.get("base_url"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        streaming=True,
        max_retries=3,
    )

def _load_persona():
    with open("config/persona.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config["persona"]

persona = _load_persona()
llm = _load_llm()

# 改为 async def
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
    
    chunks = []
    # 改为 async for 配合 astream
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
    
    full_response = "".join(chunks)
    
    return {"messages": [AIMessage(content=full_response)]}
