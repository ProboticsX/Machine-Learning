from common_imports import *

class DraftDetails(BaseModel):
    """Details about the draft"""
    summary: str = Field(description="Summary of the given topic")
    missing: str = Field(description="Missing information in the answer")
    superfluous: str = Field(description="Superfluous information in the answer")
    search_queries: list[str] = Field(description="1-3 search queries to research information and improve the answer")

class SearchResult(BaseModel):
    """Details about the search result"""
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Content of the search result")

class ExecuteToolDetails(BaseModel):
    """Details about the execution of the tool"""
    search_results: list[SearchResult] = Field(description="Search results from the tool")

class RevisorDetails(DraftDetails):
    """Details about the revisor"""
    references: list[str] = Field(description="Citations motivating your updated answer")
    revisor_count: int = Field(description="Number of times the revisor has been called")
    
class AgentState(MessagesState):
    draft_details: DraftDetails = Field(description="Details about the draft")
    execute_tool_details: ExecuteToolDetails = Field(description="Details about the execution of the tool")
    revisor_details: RevisorDetails = Field(description="Details about the revisor")
    topic: str = Field(description="The topic to research")
    first_instruction: str = Field(description="The first instruction to follow")