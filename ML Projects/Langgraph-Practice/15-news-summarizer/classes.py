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

class RAGRetrieverClass(TypedDict):
    date: str = Field(description="The date of the news in the format of YYYY-MM-DD")
    rag_retriever_results: List[Dict[str, Any]] = Field(description="The top_k relevant documents fetched from the pinecone database by the rag retriever using the tool")

class RAGGraderClass(TypedDict):
    rag_grader_final_result: str = Field(description="The binary score whether the document is relevant to the user question or not. If the document is relevant, the output should be yes. If the document is not relevant, the output should be no.")
    filtered_documents: List[Dict[str, Any]] = Field(description="The filtered documents")

class RAGGeneratorClass(TypedDict):
    rag_generator_final_answer: str = Field(description="The final answer to the user question")
    sources: List[str] = Field(description="The sources of the URLs of the final answer (in case the question is answerable)")

class RAGClass(TypedDict):
    rag_retriever_class: RAGRetrieverClass = Field(description="The rag retriever class")
    rag_grader_class: RAGGraderClass = Field(description="The rag grader class")
    rag_generator_class: RAGGeneratorClass = Field(description="The rag generator class")

class AgentState(MessagesState):
    user_question: str = Field(description="The user question")
    next: str = Field(description="The next agent to call")
    top_headlines_class: TopHeadlinesClass = Field(description="The top headlines class")
    podcast_class: PodcastClass = Field(description="The podcast class")
    category_class: CategoryClass = Field(description="The category class")
    rag_class: RAGClass = Field(description="The rag class")