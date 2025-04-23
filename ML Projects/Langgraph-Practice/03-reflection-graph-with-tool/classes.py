from common_imports import *

class ReflectionDetails(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")

class DraftDetails(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question")
    reflection: ReflectionDetails = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
    
class ReviseDetails(DraftDetails):
    references: List[str] = Field(
        description="Citations motivating your updated answer"
    )
