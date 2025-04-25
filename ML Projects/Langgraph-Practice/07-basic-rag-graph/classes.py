from common_imports import *

class AgentDetails(BaseModel):
    binary_score: str = Field(description="yes or no based on whether the question is related to US politics")

class AgentState(MessagesState):
    agent_details: AgentDetails = Field(description="Decision from the agent")
    question: str = Field(description="The question to be answered")