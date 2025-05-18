from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.newspresso_supervisor_agent import role_of_each_newspresso_supervisor_worker


def make_podcast_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_newspresso_supervisor_worker[PODCAST_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        " following workers: "+ str(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When the workers have saved the fine tuned podcast script and audio file, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def podcast_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWSPRESSO_SUPERVISOR_AGENT]]:
        print("====PODCAST SUPERVISOR NODE=====")
        print("===STATE AT PODCAST SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is some context: {context}"),
        ])
        invoke_message = {"context": last_message}
        llm_with_structured_output = llm.with_structured_output(Router)
        podcast_supervisor_chain = prompt | llm_with_structured_output
        response = podcast_supervisor_chain.invoke(invoke_message)
        print("===RESPONSE OF PODCAST SUPERVISOR NODE=====")
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
                    HumanMessage(content="The podcast audio file was generated, podcast transcript was finetuned and saved to the file.", name=PODCAST_SUPERVISOR_AGENT)
                ]
            }
           )
        return Command(goto=goto)

    return podcast_supervisor_node

role_of_each_podcast_worker = {
    PODCAST_TRANSCRIPT_SUPERVISOR_AGENT: "Supervisor agent who is tasked with managing the conversation between the workers. The goal is to first generate a podcast script and then write it to a file.",
    PODCAST_AUDIO_GENERATOR_AGENT: "Audio generator agent who is tasked with generating the audio file for the podcast and save it to a file. The audio file should be pushed to the firebase storage.",
}

# PODCAST SUPERVISOR
podcast_supervisor_members = list(role_of_each_podcast_worker.keys())
podcast_supervisor_agent = make_podcast_supervisor_node(llm, members=podcast_supervisor_members, role_of_each_worker=role_of_each_podcast_worker)