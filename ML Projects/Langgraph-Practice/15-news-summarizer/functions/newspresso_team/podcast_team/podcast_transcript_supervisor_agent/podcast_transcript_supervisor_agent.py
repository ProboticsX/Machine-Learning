from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_supervisor_agent import role_of_each_podcast_worker

def make_podcast_transcript_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_podcast_worker[PODCAST_TRANSCRIPT_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        " following workers: "+ str(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you have generated the podcast audio file, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def podcast_transcript_supervisor_node(state: AgentState) -> Command[Literal[*members, PODCAST_SUPERVISOR_AGENT]]:
        print("====PODCAST TRANSCRIPT SUPERVISOR NODE=====")
        print("===STATE AT PODCAST TRANSCRIPT SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is some context: {context}"),
        ])
        invoke_message = {"context": last_message}
        llm_with_structured_output = llm.with_structured_output(Router)
        podcast_transcript_supervisor_chain = prompt | llm_with_structured_output
        response = podcast_transcript_supervisor_chain.invoke(invoke_message)
        print("===RESPONSE OF PODCAST TRANSCRIPT SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = PODCAST_SUPERVISOR_AGENT
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return podcast_transcript_supervisor_node

role_of_each_podcast_transcript_supervisor_worker = {
    PODCAST_TRANSCRIPT_GENERATOR_AGENT: "Script generator agent who is tasked with generating a podcast script for the top headlines.",
    PODCAST_TRANSCRIPT_FINETUNER_AGENT: "Finetuner agent who is tasked with fine tuning the transcript.",
    PODCAST_TRANSCRIPT_WRITER_AGENT: "Transcript writer agent who is tasked with writing the transcript to a file.",
}

# PODCAST TRANSCRIPT SUPERVISOR
podcast_transcript_supervisor_members = list(role_of_each_podcast_transcript_supervisor_worker.keys())
podcast_transcript_supervisor_agent = make_podcast_transcript_supervisor_node(llm, members=podcast_transcript_supervisor_members, role_of_each_worker=role_of_each_podcast_transcript_supervisor_worker)