from typing import Dict, Any

from LangGraph.project_4.graph.ingestion import retriever
from LangGraph.project_4.graph.state import GraphState


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--Retrive--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}