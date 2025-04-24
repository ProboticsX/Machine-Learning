from common_imports import *

class CityDetails(BaseModel):
    """Respond to the user with this"""
    city_name: str = Field(description="Name of the city")
    state_name: str = Field(description="State name of the city")
    state_capital: str = Field(description="State capital of the city")
    country_name: str = Field(description="Country name of the city")
    country_capital: str = Field(description="Country capital of the city")

class AgentState(MessagesState):
    """State for the agent."""
    question: str = Field(description="The question to answer")
    instructions: str = Field(description="The instructions to follow")
    final_response: CityDetails = Field(description="The final response to the user")