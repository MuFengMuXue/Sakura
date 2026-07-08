from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict 

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    search_context: Optional[str]