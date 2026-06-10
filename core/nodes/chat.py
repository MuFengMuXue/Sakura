import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from core.state import ChatState
from langchain_core.messages import AIMessage

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

llm = _load_llm()

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return  {"messages": [AIMessage(content=response.content)]}