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
        
        # Create the agent with the system prompt
        news_supervisor_agent = create_react_agent(
            llm,
            tools=[],  # No tools needed for routing
            prompt=system_prompt,
            response_format=("Give me the next worker to route to next.", Router)
        )
        
        # Prepare the input
        question = state["messages"][0].content
        context = state["messages"]
        invoke_message = {
            "input": f"Here is the original user question: {question} and some context: {context}"
        }
        
        # Get the response
        response = news_supervisor_agent.invoke(invoke_message)
        
        print("===RESPONSE OF NEWS SUPERVISOR NODE=====")
        print(response)
        
        # Get the next worker from the structured response
        goto = response["structured_response"]["next"]
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