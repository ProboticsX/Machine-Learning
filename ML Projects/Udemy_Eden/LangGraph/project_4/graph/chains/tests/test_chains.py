from pprint import pprint

from dotenv import load_dotenv

from LangGraph.project_4.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from LangGraph.project_4.graph.ingestion import retriever
from LangGraph.project_4.graph.chains.generation import generation_chain

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

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)