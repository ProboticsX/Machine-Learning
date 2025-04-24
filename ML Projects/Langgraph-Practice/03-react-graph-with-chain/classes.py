from common_imports import *
class AgentState(MessagesState):
    """State for the agent."""
    question: str = Field(description="The question to answer")
    instructions: str = Field(description="The instructions to follow")

