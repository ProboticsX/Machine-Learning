from typing import Dict, Any
from xml.dom.minidom import Document

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults

from LangGraph.project_4.graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearchResults(max_results=3)
def web_search(state: GraphState) -> Dict[str, Any]:
    print("--Web Search--")
    question = state["question"]
    documents = state["documents"]
    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
