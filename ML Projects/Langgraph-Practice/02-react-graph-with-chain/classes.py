from common_imports import *
class AgentState(MessagesState):
    """State for the agent."""
    question: str = Field(description="The question to answer")

