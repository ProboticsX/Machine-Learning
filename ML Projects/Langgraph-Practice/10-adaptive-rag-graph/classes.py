from common_imports import *

class RetrieverDetails(BaseModel):
    documents: List[Document] = Field(description="List of retrieved documents")

class GraderDetails(BaseModel):
    binary_score: str = Field(description="yes or no based on whether all the documents are relevant to the question")

class GeneratorRouterDetails(BaseModel):
    binary_score: str = Field(description="yes or no based on whether the answer is grounded in the documents")

class GroundedChainDetails(BaseModel):
    binary_score: str = Field(description="yes or no based on whether the answer is correctly answers the question")

class QuestionRouterDetails(BaseModel):
    binary_score: str = Field(description="yes or no based on whether the question can be answered by the vectorstore or not")

class AgentState(MessagesState):
    retriever_details: RetrieverDetails = Field(description="Details from the retriever")
    grader_details: GraderDetails = Field(description="Details from the grader")
    generator_router_details: GeneratorRouterDetails = Field(description="Details from the generator router")
    grounded_chain_details: GroundedChainDetails = Field(description="Details from the grounded chain")
    question_router_details: QuestionRouterDetails = Field(description="Details from the question router")
    question: str = Field(description="The question to be answered")
    web_search: bool = Field(description="Whether to perform a web search")
    filtered_docs: List[Document] = Field(description="List of filtered documents")
