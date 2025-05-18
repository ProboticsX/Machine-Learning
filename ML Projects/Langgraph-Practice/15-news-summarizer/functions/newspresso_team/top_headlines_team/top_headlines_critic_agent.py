from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.helper_functions import role_of_each_random_worker

def top_headlines_critic_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUMMARIZER_AGENT]]:
    print("====TOP HEADLINES CRITIC NODE=====")
    print("===STATE AT TOP HEADLINES CRITIC NODE=====")
    print(state)

    top_headlines_summary_json_file = state["top_headlines_class"]["top_headlines_summary_json_file"]
    existing_top_headlines_critique = ""
    if state.get("top_headlines_class").get("top_headlines_critique") is not None:
        existing_top_headlines_critique = state["top_headlines_class"]["top_headlines_critique"]
    current_top_headlines_critique_count = 1
    if state.get("top_headlines_class").get("top_headlines_critique_count") is not None:
        current_top_headlines_critique_count = state["top_headlines_class"]["top_headlines_critique_count"]+1

    system_prompt = f"{role_of_each_random_worker[TOP_HEADLINES_CRITIC_AGENT]}" +"""
      Please make sure to improve the critique based on the existing critique. \n
      Example of a good critique can be such as preventing bias, adding facts, etc. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please provide the critique of the top headlines summary found in the json file: {top_headlines_summary_json_file} and the existing critique (if any): {existing_top_headlines_critique}"),
    ])
    formatted_prompt = prompt.format(top_headlines_summary_json_file=top_headlines_summary_json_file, existing_top_headlines_critique=existing_top_headlines_critique)
    top_headlines_critic_agent = create_react_agent(llm, 
                                                        tools=top_headlines_critic_tools, 
                                                        prompt = formatted_prompt)
    invoke_message = {"input": "Please critique the top headlines summary found in the json file and provide the critique."}
    result = top_headlines_critic_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES CRITIC NODE=====")
    print(result)

    current_top_headlines = TopHeadlinesClass(
            top_headlines_critique=result["messages"][-1].content,
            top_headlines_critique_count=current_top_headlines_critique_count
        )
    if state.get("top_headlines_class") is not None:
        current_top_headlines = state["top_headlines_class"].copy()
        current_top_headlines["top_headlines_critique"] = result["messages"][-1].content
        current_top_headlines["top_headlines_critique_count"] = current_top_headlines_critique_count

    return Command(
        update={
            "messages": [
                HumanMessage(content="Top headlines critiqued successfully.", name=TOP_HEADLINES_CRITIC_AGENT)
            ],
            "top_headlines_class": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUMMARIZER_AGENT,
    )