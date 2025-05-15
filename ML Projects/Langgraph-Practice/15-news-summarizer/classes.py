from common_imports import *

class TopicClass(TypedDict):
    topic: str = Field(description="The topic of the user question")

class PodcastClass(TypedDict):
    podcast_transcript: str = Field(description="The podcast transcript")

class TopHeadlinesClass(TypedDict):
    top_headlines_processed_news_file: str = Field(description="The processed news file of the top headlines")
    top_headlines_summary: str = Field(description="The summary of the top headlines")
    top_headlines_critique: str = Field(description="The critique of the top headlines")
    top_headlines_critique_count: int = Field(description="The count of the critique of the top headlines")

class AgentState(MessagesState):
    next: str = Field(description="The next agent to call")
    top_headlines: TopHeadlinesClass = Field(description="The top headlines class")
    podcast_class: PodcastClass = Field(description="The podcast class")
    topic: TopicClass = Field(description="The topic of the user question")