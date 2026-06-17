import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from core.state import ChatState

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
    )

def _load_persona():
    with open("config/persona.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["persona"]

persona = _load_persona()
llm = _load_llm()

def chat_node(state: ChatState):
    # 从状态中获取检索到的记忆上下文
    context = state.get("search_context", "")
    
    # 构建系统提示：基础人设 + 检索到的记忆
    final_system_prompt = persona
    if context:
        final_system_prompt += f"""
以下是关于**与你对话的人类用户**的历史记忆和偏好。在回答时，如果这些信息与问题相关，请优先采用它们来回答。

记忆内容：
{context}

请自然地融入这些信息，不要提及“根据记忆”等词语。
"""
    
    messages = [SystemMessage(content=final_system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}