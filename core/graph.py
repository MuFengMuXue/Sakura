from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from core.state import AgentState
from core.nodes.chat import chat_node
from core.nodes.add_memories import add_memories_node
from core.nodes.search_memories import search_memories_node
from core.tools import (memos_add_memory,memos_search_memory,memos_correct_memory)
from langchain_core.messages import AIMessage


def build_graph():
    builder = StateGraph(AgentState)
    #工具定义
    tools = [memos_add_memory,memos_search_memory,memos_correct_memory]
    tool_node = ToolNode(tools)
  
    # 注册节点
    builder.add_node("search_memories",search_memories_node)
    builder.add_node("chat", chat_node)
    builder.add_node("add_memories", add_memories_node)
    builder.add_node("tools", tool_node)

    #条件函数
    def should_continue(state: AgentState):
        messages = state.get("messages")
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"  
        else:
            return "add_memories"

    #连接图
    builder.add_edge(START,"search_memories")
    builder.add_edge("search_memories","chat")
    builder.add_conditional_edges(
        "chat",
        should_continue,
        {
            "tools": "tools",
            "add_memories": "add_memories"
        }
    )
    builder.add_edge("tools", "chat")   
    builder.add_edge("add_memories",END)
    memory_saver = MemorySaver()
    return builder.compile(checkpointer=memory_saver)

graph = build_graph()