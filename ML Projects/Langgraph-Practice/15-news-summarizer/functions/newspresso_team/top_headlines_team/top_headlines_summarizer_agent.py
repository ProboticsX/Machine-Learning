from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_summarizer_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT, TOP_HEADLINES_CRITIC_AGENT]]:
    print("====TOP HEADLINES SUMMARIZER NODE=====")
    print("===STATE AT TOP HEADLINES SUMMARIZER NODE=====")
    print(state)
    top_headlines_full_content_from_tool = state["top_headlines"]["top_headlines_full_content_from_tool"]
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
     """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the top headlines: {top_headlines_full_content_from_tool}, the critique of the top headlines: {top_headlines_critique}, the existing summary of the top headlines: {exisiting_top_headlines_summary}"),
    ])
    # llm_with_tools = llm.bind_tools(top_headlines_summarizer_tools)
    top_headlines_summarizer_chain = prompt | llm
    invoke_message = {"top_headlines_full_content_from_tool": top_headlines_full_content_from_tool, "top_headlines_critique": top_headlines_critique, "exisiting_top_headlines_summary": exisiting_top_headlines_summary}
    result = top_headlines_summarizer_chain.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES SUMMARIZER NODE=====")
    print(result)
    current_top_headlines = TopHeadlinesClass(
            top_headlines_summary=result.content
        )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_summary"] = result.content

    # If the critique count is less than the max critique count, go to the critic agent
    if current_top_headlines_critique_count < MAX_TOP_HEADLINES_CRITIQUE_COUNT:
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result.content, name=TOP_HEADLINES_SUMMARIZER_AGENT),
                    HumanMessage(content="Please provide the critique based on the summary.", name=TOP_HEADLINES_SUMMARIZER_AGENT)
                ],
                "top_headlines": current_top_headlines,
            },
            goto=TOP_HEADLINES_CRITIC_AGENT,
        )
        
    return Command(
        update={
            "messages": [
                HumanMessage(content=result.content, name=TOP_HEADLINES_SUMMARIZER_AGENT),
                HumanMessage(content="Top headlines summarized successfully.", name=TOP_HEADLINES_SUMMARIZER_AGENT)
            ],
            "top_headlines": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )