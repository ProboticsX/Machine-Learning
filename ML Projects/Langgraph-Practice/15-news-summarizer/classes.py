from common_imports import *

class TopicClass(TypedDict):
    topic: str = Field(description="The topic of the user question")

class TopHeadlinesClass(TypedDict):
    top_headlines_full_content_from_tool: str = Field(description="The full content of the top headlines from the tool")
    top_headlines_summary: str = Field(description="The summary of the top headlines")
    top_headlines_critique: str = Field(description="The critique of the top headlines")
    top_headlines_critique_count: int = Field(description="The count of the critique of the top headlines")

class AgentState(MessagesState):
    next: str = Field(description="The next agent to call")
    top_headlines: TopHeadlinesClass = Field(description="The top headlines class including the full content from the tool and the summary")
    podcast_transcript: str = Field(description="The podcast transcript")
    topic: TopicClass = Field(description="The topic of the user question")