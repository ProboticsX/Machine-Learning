from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES NODE=====")
    print("===STATE AT TOP HEADLINES NODE=====")
    print(state)
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_AGENT]}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Input: {user_request}"),
    ])
    formatted_prompt = prompt.format(user_request="Please fetch the top headlines")
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please fetch the top headlines"}
    result = top_headlines_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES NODE=====")
    print(result)
    tool_message = result["messages"][-2]
    current_top_headlines = TopHeadlinesClass(
        top_headlines_full_content_from_tool=tool_message.content
    )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_full_content_from_tool"] = tool_message.content
    return Command(
        update={
            "messages": [
                HumanMessage(content=tool_message.content, name=TOP_HEADLINES_AGENT),
                HumanMessage(content="Top headlines fetched successfully.", name=TOP_HEADLINES_AGENT)
            ],
            "top_headlines": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )