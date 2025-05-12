from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_summarizer_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES SUMMARIZER NODE=====")
    print("===STATE AT TOP HEADLINES SUMMARIZER NODE=====")
    print(state)
    top_headlines_full_content_from_tool = state["top_headlines"]["top_headlines_full_content_from_tool"]
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_SUMMARIZER_AGENT]}. The summary should be in about 200-300 words for each headline."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the top headlines: {top_headlines_full_content_from_tool}"),
    ])
    top_headlines_summarizer_chain = prompt | llm
    invoke_message = {"top_headlines_full_content_from_tool": top_headlines_full_content_from_tool}
    result = top_headlines_summarizer_chain.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES SUMMARIZER NODE=====")
    print(result)
    current_top_headlines = TopHeadlinesClass(
            top_headlines_summary=result.content
        )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_summary"] = result.content
        
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