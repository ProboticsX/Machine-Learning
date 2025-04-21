import os
import chromadb
import warnings
warnings.filterwarnings("ignore")
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain import hub
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState, START
from langchain_core.tools import tool
from typing import Literal
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import Graph
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from IPython.display import Image, display
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import PromptTemplate

from functions import generate, reflect, generate_router

load_dotenv()

workflow= StateGraph(MessagesState)

workflow.add_node("generate", generate)  # generate
workflow.add_node("reflect", reflect)  # reflect

workflow.set_entry_point("generate")
workflow.add_conditional_edges(
    "generate",
    generate_router,
    {
        "reflect": "reflect",
        END: END
    },
)
workflow.add_edge("reflect", "generate")
graph = workflow.compile()