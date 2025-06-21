from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def get_score_from_react_agent(system_prompt, top_headlines_summary_json_file, category, date, time_in_iso):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "The top headlines can be found here: {top_headlines_summary_json_file}. The category is {category} and the date is {date}. The current time in ISO format is: {time_in_iso}"),
    ])
    formatted_prompt = prompt.format(top_headlines_summary_json_file=top_headlines_summary_json_file, category=category, date=date, time_in_iso=time_in_iso)
    top_headlines_ranker_agent = create_react_agent(llm, 
                                             tools=top_headlines_ranker_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please do the task as per the system prompt"}
    return top_headlines_ranker_agent.invoke(invoke_message)


def top_headlines_ranker_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES RANKER NODE=====")
    print("===STATE AT TOP HEADLINES RANKER NODE=====")
    print(state)
    top_headlines_summary_json_file = state["top_headlines_class"]["top_headlines_summary_json_file"]
    category = state["category_class"]["category"]
    date = state["category_class"]["date"]
    current_time = get_current_time_iso()
    system_prompt_categorical_score = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_RANKER_AGENT]}"+"""
    Provides "categorical_score" which depends on the semantic value based on category for each headline. \n
    You will evaluate a news story and assign an importance score between 1 and 10. \n
    **Scoring Rules**

        - Score 10: Critical / nationally or globally significant / affects millions / deeply influential
        - Score 7-9: Major event in the category, impactful to industry or region
        - Score 4-6: Notable news, but limited impact or niche relevance
        - Score 1-3: Routine, minor updates, celebrity gossip, or soft news

        **Category-Based Adjustments**

        Business:
        - 10: Market crashes, major mergers, national budget announcements
        - 7-9: IPOs, large company quarterly earnings, new government policy
        - 1-3: Stock recommendations, local store openings

        Entertainment:
        - 10: Death of a major celebrity, global award wins, legal/criminal cases
        - 7-9: Big movie/series releases, casting of iconic roles, major interviews
        - 1-3: Relationship rumors, influencer gossip, fashion choices

        Technology:
        - 10: Global tech regulations, breakthroughs (e.g., AGI), cyber attacks
        - 7-9: Big product launches, Apple/Google/Meta news, AI trends
        - 1-3: App updates, gadget reviews

        Science:
        - 10: Major discoveries (e.g., cancer cure, fusion), Nobel wins
        - 7-9: Published breakthroughs, space missions, pandemic updates
        - 1-3: Wildlife studies, academic papers with narrow interest

        Health:
        - 10: Disease outbreaks, vaccine discoveries, mental health crisis updates
        - 7-9: New treatments, government health policy, national studies
        - 1-3: Wellness tips, diet trends

        Sports:
        - 10: World Cup/Olympic wins, doping scandals, record-breaking events
        - 7-9: Major league results, player transfers, match bans
        - 1-3: Practice updates, training news

        General:
        - 10: War, terror attack, leadership change, economic crash, large protest
        - 7-9: Policy changes, elections, protests, international diplomacy
        - 1-3: Minor accidents, local news, community events
    
    Moreover, add an entry by the name of the score: "categorical_score" for each of the top headlines you evaluate and save the file back as json in a valid format.
    """
    get_score_from_react_agent(system_prompt_categorical_score, top_headlines_summary_json_file, category, date, current_time)
    
    system_prompt_source_credibility_score = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_RANKER_AGENT]}"+"""
    Provides "source_credibility_score" which is the average of the different scores assigned per source for each headline. \n
    You need to score based on the sources attached with each top headline and assign them scores based on the given rules: \n
    **Scoring Rules**

        - Score 10: Reuters, AP, BBC
        - Score 7-9: NYT, Bloomberg, CNN, CNBC, The Verge
        - Score 4-6: Medium blog, Reddit, Instagram, Youtube
        - Score 1-3: Other websites
    
    Moreover, add an entry for the given parameter by the name of the score: "source_credibility_score" for each of the top headlines you evaluate and save the file back as json in a valid format.
    """
    get_score_from_react_agent(system_prompt_source_credibility_score, top_headlines_summary_json_file, category, date, current_time)

    system_prompt_time_sensitive_score = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_RANKER_AGENT]}"+"""
    Provides "time_sensitive_score" which is the score assigned to each headline based on how recent the news was. More recent the news and its criticality, the higher the score. You'd need to compare the published timestamp for each news against the current time\n
    You need to score each top headline and assign them scores based on the given rules: \n
    **Scoring Rules**

       - 10 = Published in last 15 minutes AND contains words like “breaking,” “alert,” “just in”
       - 7-9 = <1 hour old, notable event
       - 4-6 = any event in the past 6 hours
       - 1-3 = news published on the same day

    Moreover, add an entry for the given parameter by the name of the score: "time_sensitive_score" for each of the top headlines you evaluate and save the file back as json in a valid format.
    """
    get_score_from_react_agent(system_prompt_time_sensitive_score, top_headlines_summary_json_file, category, date, current_time)

    system_prompt_ranking_score = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_RANKER_AGENT]}"+"""
    Provides "ranking_score" which is the average of all the different scores (eg: categorical_score, source_credibility_score) for a given headline based on the weights assigned.\n
    Moreover, add an entry for the given parameter by the name of the score: "ranking_score" for each of the top headlines you evaluate and save the file back as json in a valid format.
    """
    result = get_score_from_react_agent(system_prompt_ranking_score, top_headlines_summary_json_file, category, date, current_time)
    print("===RESULT OF TOP HEADLINES RANKER NODE=====")
    print(result)

    image_url_verified_string = "The image URL of the top headlines are not verified yet."
    if category == "general":
        image_url_verified_string = "The image URL of the top headlines are verified successfully."
    return Command(
        update={
            "messages": [
                HumanMessage(content=f"Top headlines fetched and summarized successfully. The headlines were ranked successfully. {image_url_verified_string}.", name=TOP_HEADLINES_RANKER_AGENT)
            ],
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )