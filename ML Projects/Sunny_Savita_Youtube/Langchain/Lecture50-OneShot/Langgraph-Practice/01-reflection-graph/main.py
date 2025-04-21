import os
import chromadb
import warnings
import json
from datetime import datetime
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

from graph import graph
from functions import displayGraph

load_dotenv()

class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            # Convert the object to a dictionary
            d = obj.__dict__.copy()
            # Remove any non-serializable attributes if needed
            if 'lc_kwargs' in d:
                d['lc_kwargs'] = str(d['lc_kwargs'])
            return d
        return super().default(obj)

if __name__ == "__main__":
    print("Hello from Reflection Graph!")
    tweet = """ @LangChainAI — newly Tool Calling feature is seriously underrated. \n
                After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy. \n
                Made a video covering their newest blog post"""
    displayGraph(graph)
    response = graph.invoke({"messages":[tweet]})
    
    # Store complete message objects
    messages_data = {
        "messages": response['messages'],
        "timestamp": datetime.now().isoformat()
    }
    
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reflection_response_{timestamp}.json"
    
    # Save to JSON file with custom encoder
    with open(output_file, 'w') as f:
        json.dump(messages_data, f, indent=2, cls=MessageEncoder)
    
    print(f"\nResponse saved to: {output_file}")
    print("\nMessages:")
    for msg in response['messages']:
        msg.pretty_print()