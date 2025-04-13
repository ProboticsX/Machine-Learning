import os
import warnings
warnings.filterwarnings("ignore")
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

load_dotenv()
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}"),
    ("user", "{text}")
])

llm = ChatOpenAI()
parser = StrOutputParser()
chain = prompt_template | llm | parser

app = FastAPI(
    title="My API",
    description="My first LLM API",
    version="1.0"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)