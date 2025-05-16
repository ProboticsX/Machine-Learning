from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.news_summarizer_agent import role_of_each_news_supervisor_worker

def make_top_headlines_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_news_supervisor_worker[TOP_HEADLINES_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        " following workers: "+ str(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you have received the summary of the top headlines, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def top_headlines_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWS_SUPERVISOR_AGENT]]:
        print("====TOP HEADLINES SUPERVISOR NODE=====")
        print("===STATE AT TOP HEADLINES SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is some context: {context}"),
        ])
        invoke_message = {"context": last_message}
        llm_with_structured_output = llm.with_structured_output(Router)
        top_headlines_supervisor_chain = prompt | llm_with_structured_output
        response = top_headlines_supervisor_chain.invoke(invoke_message)
        print("===RESPONSE OF TOP HEADLINES SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        print("===GOTO=====")
        print(goto)
        if goto == "FINISH":
            goto = NEWS_SUPERVISOR_AGENT
            return Command(
            goto=goto, 
            update={
                "next": goto,
                "messages": [
                    HumanMessage(content="The top headlines were generated, summarized and saved to the file.", name=TOP_HEADLINES_SUPERVISOR_AGENT)
                ]
            }
           )
        return Command(goto=goto, update={"next": goto})

    return top_headlines_supervisor_node




role_of_each_top_headlines_worker = {
    TOP_HEADLINES_AGENT: "Agent who is tasked with providing the top headlines with full content.",
    TOP_HEADLINES_SUMMARIZER_AGENT: "Agent who is tasked with summarizing the top headlines.",
}

# TOP HEADLINES SUPERVISOR
top_headlines_members = list(role_of_each_top_headlines_worker.keys())
top_headlines_supervisor_agent = make_top_headlines_supervisor_node(llm, members=top_headlines_members, role_of_each_worker=role_of_each_top_headlines_worker)
