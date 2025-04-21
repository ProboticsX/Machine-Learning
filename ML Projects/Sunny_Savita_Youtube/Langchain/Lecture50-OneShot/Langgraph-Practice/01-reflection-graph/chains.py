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


load_dotenv()

def get_llm():
    return ChatOpenAI()

def get_generate_chain(state):
    user_tweet = state["messages"][0].content
    system_message = """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
                Generate the best twitter post possible for the user's request.
                If the user provides critique, respond with a revised version of your previous attempts."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "Here's the user's tweet: {user_tweet}"),
        ]
    )
    generate_chain = prompt | get_llm()
    return generate_chain, user_tweet

def get_reflect_chain(state):
    last_tweet = state["messages"][-1].content
    system_message = """You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
            Always provide detailed recommendations, including requests for length, virality, style, etc."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "Here's the last tweet: {last_tweet}"),
        ]
    )
    reflect_chain = prompt | get_llm()
    return reflect_chain, last_tweet