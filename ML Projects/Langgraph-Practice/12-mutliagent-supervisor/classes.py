from common_imports import *

class AgentState(MessagesState):
    question: str = Field(description="The question to be answered")