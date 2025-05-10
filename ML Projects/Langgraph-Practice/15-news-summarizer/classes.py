from common_imports import *


class TopHeadlinesClass(TypedDict):
    top_headlines_full_content_from_tool: str = Field(description="The full content of the top headlines from the tool")
    top_headlines_summary: str = Field(description="The summary of the top headlines")

class AgentState(MessagesState):
    next: str = Field(description="The next agent to call")
    top_headlines: TopHeadlinesClass = Field(description="The top headlines class including the full content from the tool and the summary")