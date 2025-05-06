from common_imports import *

class AgentState(MessagesState):
    next: str = Field(description="The next agent to call")