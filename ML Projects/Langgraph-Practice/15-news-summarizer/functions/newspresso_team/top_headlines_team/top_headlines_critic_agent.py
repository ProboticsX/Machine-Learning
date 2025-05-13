from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.helper_functions import role_of_each_random_worker

def top_headlines_critic_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUMMARIZER_AGENT]]:
    print("====TOP HEADLINES CRITIC NODE=====")
    print("===STATE AT TOP HEADLINES CRITIC NODE=====")
    print(state)

    top_headlines_summary = state["top_headlines"]["top_headlines_summary"]
    existing_top_headlines_critique = ""
    if state.get("top_headlines").get("top_headlines_critique") is not None:
        existing_top_headlines_critique = state["top_headlines"]["top_headlines_critique"]
    current_top_headlines_critique_count = 0
    if state.get("top_headlines").get("top_headlines_critique_count") is not None:
        current_top_headlines_critique_count = state["top_headlines"]["top_headlines_critique_count"]+1

    system_prompt = f"{role_of_each_random_worker[TOP_HEADLINES_CRITIC_AGENT]}" +"""
      Please make sure to improve the critique based on the existing critique. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please provide the critique of the top headlines summary: {top_headlines_summary} and the existing critique: {existing_top_headlines_critique}"),
    ])
    top_headlines_critic_chain = prompt | llm
    invoke_message = {"top_headlines_summary": top_headlines_summary, "existing_top_headlines_critique": existing_top_headlines_critique}
    result = top_headlines_critic_chain.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES CRITIC NODE=====")
    print(result)

    current_top_headlines = TopHeadlinesClass(
            top_headlines_critique=result.content,
            top_headlines_critique_count=current_top_headlines_critique_count
        )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_critique"] = result.content
        current_top_headlines["top_headlines_critique_count"] = current_top_headlines_critique_count

    return Command(
        update={
            "messages": [
                HumanMessage(content=result.content, name=TOP_HEADLINES_CRITIC_AGENT),
                HumanMessage(content="Top headlines critiqued successfully.", name=TOP_HEADLINES_CRITIC_AGENT)
            ],
            "top_headlines": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUMMARIZER_AGENT,
    )