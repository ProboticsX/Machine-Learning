from dotenv import load_dotenv

from LangGraph.project_4.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from LangGraph.project_4.graph.ingestion import retriever

load_dotenv()
def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(doc_txt)
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(doc_txt)
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"