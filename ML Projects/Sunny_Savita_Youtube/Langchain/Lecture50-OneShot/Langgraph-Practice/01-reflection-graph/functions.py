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

from chains import get_generate_chain, get_reflect_chain
load_dotenv()

def displayGraph(graph):
    graph.get_graph().draw_mermaid_png(output_file_path="reflection_graph.png")

def generate(state):
    print("===GENERATE===")
    generate_chain, user_tweet = get_generate_chain(state)
    response = generate_chain.invoke({"user_tweet": user_tweet})
    return {"messages": [response]}

def reflect(state):
    print("===REFLECT===")
    reflect_chain, last_tweet = get_reflect_chain(state)
    response = reflect_chain.invoke({"last_tweet": last_tweet})
    return {"messages": [response]}

def generate_router(state):
    if len(state["messages"]) > 6:
        return END
    return "reflect"

