from common_imports import *

class CityDetails(BaseModel):
    """Respond to the user with this"""
    city_name: str = Field(description="Name of the city")
    state_name: str = Field(description="State name of the city")
    state_capital: str = Field(description="State capital of the city")
    country_name: str = Field(description="Country name of the city")
    country_capital: str = Field(description="Country capital of the city")
    city_summary: str = Field(description="Summary of the city in 2-3 sentences")

class OutputRouterState(BaseModel):
    """State for the output router."""
    binary_score: int = Field(description="Indicates if the user's question is related to the city/geographical location or not")

class AgentState(MessagesState):
    """State for the agent."""
    question: str = Field(description="The question to answer")
    instructions: str = Field(description="The instructions to follow")
    final_response: CityDetails = Field(description="The final response to the user")
    output_router_state: OutputRouterState = Field(description="The state for the output router")