from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_summarizer_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT, TOP_HEADLINES_CRITIC_AGENT]]:
    print("====TOP HEADLINES SUMMARIZER NODE=====")
    print("===STATE AT TOP HEADLINES SUMMARIZER NODE=====")
    print(state)
    top_headlines_processed_news_file = state["top_headlines"]["top_headlines_processed_news_file"]
    top_headlines_critique = ""
    if state.get("top_headlines").get("top_headlines_critique") is not None:
        top_headlines_critique = state["top_headlines"]["top_headlines_critique"]
    exisiting_top_headlines_summary = ""
    if state.get("top_headlines").get("top_headlines_summary") is not None:
        exisiting_top_headlines_summary = state["top_headlines"]["top_headlines_summary"]
    current_top_headlines_critique_count = 0
    if state.get("top_headlines").get("top_headlines_critique_count") is not None:
        current_top_headlines_critique_count = state["top_headlines"]["top_headlines_critique_count"]
    
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_SUMMARIZER_AGENT]}."+"""
      The summary should be in about 200-300 words for each headline. \n
      Please make sure to improve the summary based on the critique of the top headlines and the existing summary. \n
      Moreover, write the summary to a text file along with the relevant links/sources to the various top headlines in the summary. For example:\n
       - <Top Headline 1 Summary> \n
           - Relevant links: [Link 1, Link 2, Link 3] \n
       - <Top Headline 2 Summary> \n
           - Relevant links: [Link 1, Link 2, Link 3] \n
       - <Top Headline 3 Summary> \n
           - Relevant links: [Link 1, Link 2, Link 3] \n
    Additionally, write the summary to a json file where each json object contains the following fields, make sure there are no invalid control characters in the json file:\n
        - source: <Source of the top headline> \n
        - author: <Author of the top headline> \n
        - title: <Title of the top headline> \n
        - description: <Description of the top headline> \n
        - url: <URL of the top headline> \n
        - urlToImage: <URL to the image of the top headline> \n
        - published_at: <Published at of the top headline> \n
        - content_summary: <summary of the top headline content which is the same as generated above> \n
    Lastly, push the summary to the firebase database.
     """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the file where you can find the top headlines: {top_headlines_processed_news_file}, the critique of the top headlines: {top_headlines_critique}, the existing summary of the top headlines: {exisiting_top_headlines_summary}"),
    ])
    formatted_prompt = prompt.format(top_headlines_processed_news_file=top_headlines_processed_news_file, top_headlines_critique=top_headlines_critique, exisiting_top_headlines_summary=exisiting_top_headlines_summary)
    top_headlines_summarizer_agent = create_react_agent(llm, 
                                                        tools=top_headlines_summarizer_tools, 
                                                        prompt = formatted_prompt)
    result = top_headlines_summarizer_agent.invoke({"input": "Please summarize the top headlines and write the summary to a text file and json file."})
    print("===RESULT OF TOP HEADLINES SUMMARIZER NODE=====")
    print(result)
    current_top_headlines = TopHeadlinesClass(
            top_headlines_summary=result["messages"][-1].content
        )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_summary"] = result["messages"][-1].content

    # If the critique count is less than the max critique count, go to the critic agent
    if current_top_headlines_critique_count < MAX_TOP_HEADLINES_CRITIQUE_COUNT:
        return Command(
            update={
                "messages": [
                    HumanMessage(content="Please provide the critique based on the summary.", name=TOP_HEADLINES_SUMMARIZER_AGENT)
                ],
                "top_headlines": current_top_headlines,
            },
            goto=TOP_HEADLINES_CRITIC_AGENT,
        )
        
    return Command(
        update={
            "messages": [
                HumanMessage(content="Top headlines summarized successfully! The summary has been written to a text file and a json file and pushed to the firebase database.", name=TOP_HEADLINES_SUMMARIZER_AGENT)
            ],
            "top_headlines": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )