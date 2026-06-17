from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from core.state import ChatState
from core.nodes.chat import chat_node
from core.nodes.search_memories_http import search_memories_node
from core.nodes.add_memories_http import add_memories_node
def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_node("search_memories", search_memories_node)
    builder.add_node("add_memories",add_memories_node)
    builder.add_edge(START, "search_memories")
    builder.add_edge("search_memories","chat")
    builder.add_edge("chat","add_memories")
    builder.add_edge("add_memories", END)
    memory_saver = MemorySaver()
    return builder.compile(checkpointer=memory_saver)

graph = build_graph()