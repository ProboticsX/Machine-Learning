import os
import chromadb
import warnings
import json
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain import hub
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState, START, Graph, MessageGraph
from langchain_core.tools import tool, InjectedToolCallId
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import tools_condition, InjectedState
from langchain_core.prompts import PromptTemplate
from constants import *
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.types import Command
from langgraph_supervisor import create_supervisor
from typing import Annotated

# Other imports
from pydantic import BaseModel, Field
from IPython.display import Image, display
from typing import List, Literal
from typing_extensions import TypedDict
import yfinance as yf
import math
from newsapi import NewsApiClient
from serpapi import GoogleSearch
from tools.news_api_tool.news_api_tool import NewsTool
from podcastfy.client import generate_podcast


# Load environment variables
load_dotenv()

llm = ChatOpenAI(model_name=MODEL_NAME)