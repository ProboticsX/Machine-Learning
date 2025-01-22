import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub


load_dotenv()
pythonpath = os.getenv('PYTHONPATH')
if pythonpath and pythonpath not in sys.path:
    sys.path.append(pythonpath)
    print(sys.path)

from Langchain.project_1.tools.tools import get_profile_url_tavily

def lookup_twitter(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
    )
    template = """Given the full name {name_of_person} I want you to get it me a link to their Twitter profile page.
                    Your answer should only contain the user's Twitter username. Remove @ from the username.
    """
    prompt_template = PromptTemplate(input_variables=["name_of_person"], template=template)
    tools_for_agent = [
        Tool(
            name = "Crawl Google for Twitter Profile Page",
            func = get_profile_url_tavily,
            description = "useful when you need to get the Twitter Profile Page URL"
        )
    ]
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt) #recipe we send to agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    result = agent_executor.invoke(
        input={"input":prompt_template.format_prompt(name_of_person=name)}
    )
    twitter_profile_url = result["output"]
    return twitter_profile_url

if __name__ == "__main__":
    twitter_url = lookup_twitter(name="Elon Musk")
    print(twitter_url)