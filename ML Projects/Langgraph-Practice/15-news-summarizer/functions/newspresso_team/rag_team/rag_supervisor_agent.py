from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.newspresso_supervisor_agent import role_of_each_newspresso_supervisor_worker

def make_rag_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_newspresso_supervisor_worker[RAG_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        " following workers: "+ str(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you have received the answer to the RAG related user question, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def rag_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWSPRESSO_SUPERVISOR_AGENT]]:
        print("====RAG SUPERVISOR NODE=====")
        print("===STATE AT RAG SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is some context: {context}"),
        ])
        invoke_message = {"context": last_message}
        llm_with_structured_output = llm.with_structured_output(Router)
        rag_supervisor_chain = prompt | llm_with_structured_output
        response = rag_supervisor_chain.invoke(invoke_message)
        print("===RESPONSE OF RAG SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        print("===GOTO=====")
        print(goto)
        if goto == "FINISH":
            goto = NEWSPRESSO_SUPERVISOR_AGENT
            return Command(
            goto=goto, 
            update={
                "next": goto,
                "messages": [
                    HumanMessage(content="The RAG related user question was answered successfully.", name=RAG_SUPERVISOR_AGENT)
                ]
            }
           )
        return Command(goto=goto, update={"next": goto})

    return rag_supervisor_node




role_of_each_rag_worker = {
    RAG_RETRIEVER_AGENT: "Agent who is tasked with retrieving the top_k relevant documents from the pinecone database according to the user question.",
}

# RAG AGENT SUPERVISOR
rag_supervisor_agent_members = list(role_of_each_rag_worker.keys())
rag_supervisor_agent = make_rag_supervisor_node(llm, members=rag_supervisor_agent_members, role_of_each_worker=role_of_each_rag_worker)
