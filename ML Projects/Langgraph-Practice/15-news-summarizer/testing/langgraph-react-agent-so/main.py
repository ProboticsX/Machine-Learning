from classes import BooleanClass
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent, ToolNode
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o-mini")

system_prompt = """You need to reply in either True or False"""
user_question = "Is the sky blue?"
formatted_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_question)
])

top_headlines_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=formatted_prompt,
    response_format=BooleanClass,
)

invoke_message = {"input": "Please do the task as per the system prompt"}
result = top_headlines_agent.invoke(invoke_message)
print(result)
