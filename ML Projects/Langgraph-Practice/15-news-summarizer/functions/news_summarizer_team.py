from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *

def make_news_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: "+ ", ".join(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you think you have answered the user request, respond with FINISH."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def news_supervisor_node(state: AgentState) -> Command[Literal[*members, END]]:
        print("====NEWS SUPERVISOR NODE=====")
        print("===STATE AT NEWS SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        context = state["messages"]
        question = state["messages"][0].content
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is the original user question: {question} and some context: {context}"),
        ])
        invoke_message = {"context": context, "question": question}
        llm_with_structured_output = llm.with_structured_output(Router)
        news_supervisor_chain = prompt | llm_with_structured_output
        response = news_supervisor_chain.invoke(invoke_message)
        print("===RESPONSE OF NEWS SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return news_supervisor_node


role_of_each_news_supervisor_worker = {
    PODCAST_SUPERVISOR_AGENT: "Supervisor agent who is tasked with providing the podcast script and audio file.",
    TOP_HEADLINES_SUPERVISOR_AGENT: "Supervisor agent who is tasked with providing the top headlines along with the summary of the news."
}


#NEWS SUPERVISOR
news_supervisor_members = list(role_of_each_news_supervisor_worker.keys())
news_supervisor_agent = make_news_supervisor_node(llm, members=news_supervisor_members, role_of_each_worker=role_of_each_news_supervisor_worker)