from common_imports import *

class CityDetails(BaseModel):
    """Respond to the user with this"""
    city_name: str = Field(description="Name of the city")
    state_name: str = Field(description="State name of the city")
    state_capital: str = Field(description="State capital of the city")
    country_name: str = Field(description="Country name of the city")
    country_capital: str = Field(description="Country capital of the city")
    summary: str = Field(description="Summary of the city")

class ResearchResponse(BaseModel):
    """Response format for research agent"""
    findings: str = Field(description="The research findings")
    sources: List[str] = Field(description="List of sources used")
    summary: str = Field(description="Brief summary of the findings")

class MathResponse(BaseModel):
    """Response format for math agent"""
    calculation: str = Field(description="The mathematical calculation performed")
    result: float = Field(description="The numerical result")
    explanation: str = Field(description="Explanation of the calculation")

class FinanceResponse(BaseModel):
    """Response format for finance agent"""
    ticker: str = Field(description="The stock ticker symbol")
    price: float = Field(description="The stock price")
    company_name: str = Field(description="Name of the company")
    additional_info: str = Field(description="Any additional financial information")

class AgentState(MessagesState):
    """State for the agent."""
    question: str = Field(description="The question to answer")
    instructions: str = Field(description="The instructions to follow")
    final_response: CityDetails = Field(description="The final response to the user")