from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *

def make_newspresso_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: "+ ", ".join(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a task and respond with their results and status." 
        "When you think you have answered the user request, respond with FINISH."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def newspresso_supervisor_node(state: AgentState) -> Command[Literal[*members, END]]:
        print("====NEWSPRESSO SUPERVISOR NODE=====")
        print("===STATE AT NEWSPRESSO SUPERVISOR NODE=====")
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
        print("===RESPONSE OF NEWSPRESSO SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return newspresso_supervisor_node


role_of_each_newspresso_supervisor_worker = {
    PODCAST_SUPERVISOR_AGENT: "Supervisor agent who is tasked with providing the fine tuned podcast script and audio file. The goal is to first generate a fine tuned podcast transcript and then write it to a file. Lastly, the audio file should be generated from the fine tuned transcript.",
    TOP_HEADLINES_SUPERVISOR_AGENT: "Supervisor agent who is tasked with providing the top headlines along with the summary of the news. The goal is to fetch the top headlines and summarize the news. Moreover, the summary is written to a json file and pushed to the firebase database."
}


#NEWS SUPERVISOR
newspresso_supervisor_members = list(role_of_each_newspresso_supervisor_worker.keys())
newspresso_supervisor_agent = make_newspresso_supervisor_node(llm, members=newspresso_supervisor_members, role_of_each_worker=role_of_each_newspresso_supervisor_worker)