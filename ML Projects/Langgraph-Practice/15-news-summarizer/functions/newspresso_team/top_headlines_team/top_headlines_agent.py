from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass, CategoryClass
from tools.helper_tools.tools import *
from tools.helper_tools.tools import get_perplexity_payload, get_perplexity_headers
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES NODE=====")
    print("===STATE AT TOP HEADLINES NODE=====")
    print(state)

    def category_extractor(question: str) -> str:
        """Get the category from the user request."""
        system_prompt = """You are given a user question and you need to extract the category, country and date from it.
                       These are the categories you can choose from: \n
                        - business \n
                        - technology \n
                        - science \n
                        - health \n
                        - entertainment \n
                        - sports \n
                        - general (choose this if cannot find a specific category) \n

                        Also, find out what country the user is from. If not mentioned, choose USA. \n
                        Finally, output the today's date in the format of YYYY-MM-DD.
                       """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", question),
        ])
        formatted_prompt = prompt.format(question=question)
        category_extractor_agent = create_react_agent(llm, tools=category_extractor_agent_tools, prompt=formatted_prompt, response_format=CategoryClass)
        invoke_message = {"input": "Please do the task as per the system prompt"}
        result = category_extractor_agent.invoke(invoke_message)
        print("===RESULT OF CATEGORY EXTRACTOR AGENT=====")
        print(result)
        return result["structured_response"]["category"], result["structured_response"]["country"], result["structured_response"]["date"]

    user_question = state["messages"][0].content
    category_class = category_extractor(user_question)
    category = category_class[0]
    country = category_class[1]
    date = category_class[2]
    print("===CATEGORY, COUNTRY AND DATE=====")
    print(category, country, date)
    payload = get_perplexity_payload(f"What are the latest top 5 headlines in the {category} category in {country} on {date}?")
    headers = get_perplexity_headers()
    response = requests.request("POST", PERPLEXITY_API_URL, json=payload, headers=headers)
    top_headlines = response.json()["choices"][0]["message"]["content"]
    print("===BASIC TOP HEADLINES FROM PERPLEXITY API=====")
    print(top_headlines)
    
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_AGENT]}"+"""
    You are given a list of top headlines. \n
    You need to enrich each of the headlines with the following information using the tool provided. \n
    Make sure that the enriched content is relevant to the top headlines provided. \n
    Please do not skip any of the headlines. \n
    Save the results in a json file and the format for each headline should be as follows: \n
        - title: The title of the headline.\n
        - description: Short description of the headline. It should be around 100 words.\n
        - content_summary: A summary of the headline in about 300 words.\n
        - urlToImage: The image URL of the headline.\n
        - sources: The list of citations/relevant sources of the headline. Can be around 2-3 sources per headline.\n
        - published_at: Today's date \n
    Lastly, make sure the json file has a valid JSON format. Do not include any control characters in the json file. If not, you need to fix it and then save it back.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please enrich the following top headlines with the information provided: {top_headlines}. The summary json file path is {summary_json_file_path}"),
    ])
    summary_json_file_path = summary_json_file
    formatted_prompt = prompt.format(top_headlines=top_headlines, summary_json_file_path=summary_json_file_path)
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please do the task as per the system prompt"}
    result = top_headlines_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES NODE=====")
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
                HumanMessage(content="Top headlines fetched and summarized successfully. The json file was saved but not pushed to the firebase database. The image URL of the top headlines are not verified yet.", name=TOP_HEADLINES_AGENT)
            ],
            "top_headlines_class": current_top_headlines,
            "category_class": CategoryClass(category=category, country=country, date=date),
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )