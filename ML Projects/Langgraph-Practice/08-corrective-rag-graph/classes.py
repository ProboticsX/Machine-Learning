from common_imports import *

class RetrieverDetails(BaseModel):
    documents: List[Document] = Field(description="List of retrieved documents")

class GraderDetails(BaseModel):
    binary_score: str = Field(description="yes or no based on whether all the documents are relevant to the question")

class AgentState(MessagesState):
    retriever_details: RetrieverDetails = Field(description="Details from the retriever")
    grader_details: GraderDetails = Field(description="Details from the grader")
    question: str = Field(description="The question to be answered")
    web_search: bool = Field(description="Whether to perform a web search")
    filtered_docs: List[Document] = Field(description="List of filtered documents")