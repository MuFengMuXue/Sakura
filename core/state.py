from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict 

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    search_context: Optional[str]
    memory_buffer : List[tuple[str,str]] = []
    buffer_count: int = 0