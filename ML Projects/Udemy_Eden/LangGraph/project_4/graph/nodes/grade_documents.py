from typing import Dict, Any

from dotenv import load_dotenv

from LangGraph.project_4.graph.chains.retrieval_grader import retrieval_grader
from LangGraph.project_4.graph.state import GraphState

load_dotenv()
def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("--GRADE: RELEVANT DOCUMENT--")
            filtered_docs.append(d)
        else:
            print("--GRADE: IRRELEVANT DOCUMENT--")
            web_search = True
    return {"documents": filtered_docs, "question": question, "web_search": web_search}