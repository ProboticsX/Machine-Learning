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

    def get_topic_from_user_question(question: str) -> str:
        """Get the topic from the user request."""
        system_prompt = """You are given a user question and you need to extract the topic from it.
                       These are the topics you can choose from: \n
                        - business \n
                        - technology \n
                        - science \n
                        - health \n
                        - entertainment \n
                        - sports \n
                        - general (choose this if cannot find a specific topic) \n
                       """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here's the user question: {question}"),
        ])
        llm_with_structured_output = llm.with_structured_output(CategoryClass)
        topic_extractor_chain = prompt | llm_with_structured_output
        invoke_message = {"question": question}
        result = topic_extractor_chain.invoke(invoke_message)
        print("===RESULT OF TOPIC EXTRACTOR CHAIN=====")
        print(result)
        return result["category"]

    user_question = state["messages"][0].content
    category = get_topic_from_user_question(user_question)
    payload = get_perplexity_payload(f"What are the top 5 headlines in the {category} category for today?")
    headers = get_perplexity_headers()
    response = requests.request("POST", PERPLEXITY_API_URL, json=payload, headers=headers)
    top_headlines = response.json()["choices"][0]["message"]["content"]
    print("===BASIC TOP HEADLINES FROM PERPLEXITY API=====")
    print(top_headlines)
    
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_AGENT]}"+"""
    You are given a list of top headlines. \n
    You need to enrich each of the headlines with the following information using the tool provided. \n
    Make sure that the enriched content is relevant to the top headlines provided. \n
    Save the results in a json file and the format for each headline should be as follows: \n
        - title: The title of the headline.\n
        - description: Short description of the headline. It should be around 100 words.\n
        - content_summary: A summary of the headline in about 300 words.\n
        - urlToImage: The image URL of the headline.\n
        - sources: The list of citations/relevant sources of the headline. Can be around 2-3 sources per headline.\n
        - published_at: Today's date
    Lastly, push the json file to the firebase database as per the category.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please enrich the following top headlines with the information provided: {top_headlines}"),
    ])
    formatted_prompt = prompt.format(top_headlines=top_headlines)
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please enrich the top headlines with the information provided. Make sure not to skip any of the headlines. Save the results in a json file and push it to the firebase database."}
    result = top_headlines_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES NODE=====")
    print(result)

    current_top_headlines = TopHeadlinesClass(
        top_headlines_processed_news_file=processsed_news_file_path
    )
    if state.get("top_headlines_class") is not None:
        current_top_headlines = state["top_headlines_class"].copy()
        current_top_headlines["top_headlines_processed_news_file"] = processsed_news_file_path
    return Command(
        update={
            "messages": [
                HumanMessage(content="Top headlines fetched successfully. The json file was saved and has been pushed to the firebase database.", name=TOP_HEADLINES_AGENT)
            ],
            "top_headlines_class": current_top_headlines,
            "category_class": CategoryClass(category=category),
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )