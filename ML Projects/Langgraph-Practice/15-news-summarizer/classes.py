from common_imports import *
from typing import List, Dict, Any

class CategoryClass(TypedDict):
    category: str = Field(description="The category of the news")
    country: str = Field(description="The country of the news")
    date: str = Field(description="The date of the news in the format of YYYY-MM-DD")

class PodcastClass(TypedDict):
    podcast_transcript_file_path: str = Field(description="The path of the podcast transcript file")
    podcast_transcript_critique: str = Field(description="The critique of the podcast transcript")
    podcast_transcript_critique_count: int = Field(description="The count of the critique of the podcast transcript")

class TopHeadlinesClass(TypedDict):
    top_headlines_processed_news_file: str = Field(description="The processed news file of the top headlines")
    top_headlines_summary_json_file: str = Field(description="The json file of the top headlines summary")
    top_headlines_critique: str = Field(description="The critique of the top headlines")
    top_headlines_critique_count: int = Field(description="The count of the critique of the top headlines")

class RAGClass(TypedDict):
    date: str = Field(description="The date of the news in the format of YYYY-MM-DD")
    rag_retriever_results: List[Dict[str, Any]] = Field(description="The results of the rag retriever")

class AgentState(MessagesState):
    next: str = Field(description="The next agent to call")
    top_headlines_class: TopHeadlinesClass = Field(description="The top headlines class")
    podcast_class: PodcastClass = Field(description="The podcast class")
    category_class: CategoryClass = Field(description="The category class")