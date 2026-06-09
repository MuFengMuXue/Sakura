from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from core.state import ChatState
from core.nodes.chat import chat_node

def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    memory_saver = MemorySaver()
    return builder.compile(checkpointer=memory_saver)

app = build_graph()