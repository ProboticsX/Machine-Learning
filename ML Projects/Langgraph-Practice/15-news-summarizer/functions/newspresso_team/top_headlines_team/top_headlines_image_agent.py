from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_image_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES IMAGE NODE=====")
    print("===STATE AT TOP HEADLINES IMAGE NODE=====")
    print(state)

    top_headlines_summary_json_file = state["top_headlines_class"]["top_headlines_summary_json_file"]

    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_IMAGE_AGENT]}"+"""
    You are given a json file that contains the top headlines which will include the image URL (urlToImage parameter in the json object) for each headline. \n
    You need to check if the image URL of the top headlines is valid and points to an actual image file. \n
    If the image URL is not valid, you need to find a new image URL related to the headline (content_summary parameter in the json object) and update the json file. \n
    If the image URL is valid, you need to check if the image can be opened. \n
    If the image cannot be opened, you need to find a new image URL related to the headline and update the json file. \n
    If the image can be opened, no update for that headline is needed as the image URL is valid. \n
    Lastly, make sure the json file has a valid JSON format. If not, you need to fix it and then save it back.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please verify the image URL of the each of the top headlines and update the json file if needed: {top_headlines_summary_json_file}"),
    ])
    formatted_prompt = prompt.format(top_headlines_summary_json_file=top_headlines_summary_json_file)
    top_headlines_image_agent = create_react_agent(llm, 
                                             tools=top_headlines_image_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please do the task as per the system prompt"}
    result = top_headlines_image_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES IMAGE NODE=====")
    print(result)

    current_top_headlines = TopHeadlinesClass(
            top_headlines_summary_json_file=summary_json_file
    )
    if state.get("top_headlines_class") is not None:
        current_top_headlines = state["top_headlines_class"].copy()
        current_top_headlines["top_headlines_summary_json_file"] = summary_json_file
    return Command(
        update={
            "messages": [
                HumanMessage(content="Top headlines fetched and summarized successfully. The image URLs of the top headlines were verified and updated if needed. The json file is not pushed to the firebase database yet.", name=TOP_HEADLINES_IMAGE_AGENT)
            ],
            "top_headlines_class": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )