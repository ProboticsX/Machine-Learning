from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent

load_dotenv()

react_prompt: PromptTemplate = hub.pull("hwchase17/react")

@tool
def triple(num: float) -> float:
    """
    :param num: a numbe to triple
    :return: the number tripled: multiply by 3
    """
    return 3*float(num)

tools = [TavilySearchResults(max_results=1), triple]
llm = ChatOpenAI(model_name="gpt-4o-mini")
react_agent_runnable = create_react_agent(llm, tools, react_prompt)