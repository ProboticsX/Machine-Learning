from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_firebase_pusher_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES FIREBASE PUSHER NODE=====")
    print("===STATE AT TOP HEADLINES FIREBASE PUSHER NODE=====")
    print(state)
    top_headlines_summary_json_file = state["top_headlines_class"]["top_headlines_summary_json_file"]
    category = state["category_class"]["category"]
    date = state["category_class"]["date"]
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_FIREBASE_PUSHER_AGENT]}"+"""
    You are given a json file that contains the top headlines. Make sure the json file has a valid JSON format. Do not include any control characters in the json file. \n
    If not in the correct format, you need to fix it and then save it back.\n
    You need to push the valid json file to the firebase database. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please push the json file to the firebase database: {top_headlines_summary_json_file}. The category is {category} and the date is {date}"),
    ])
    formatted_prompt = prompt.format(top_headlines_summary_json_file=top_headlines_summary_json_file, category=category, date=date)
    top_headlines_firebase_pusher_agent = create_react_agent(llm, 
                                             tools=top_headlines_firebase_pusher_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please do the task as per the system prompt"}
    result = top_headlines_firebase_pusher_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES FIREBASE PUSHER NODE=====")
    print(result)

    return Command(
        update={
            "messages": [
                HumanMessage(content="Top headlines fetched and summarized successfully. The image URLs of the top headlines were verified and pushed to the firestore database.", name=TOP_HEADLINES_FIREBASE_PUSHER_AGENT)
            ],
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )