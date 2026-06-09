from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict 

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]