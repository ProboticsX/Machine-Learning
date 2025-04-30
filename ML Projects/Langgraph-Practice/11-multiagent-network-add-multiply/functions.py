from common_imports import *
from constants import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-network-add-multiply.png")

def addition_agent_function(state) -> Command[Literal[MULTIPLICATION_AGENT, END]]:
    print("==========ADDITION AGENT==========")
    print(state)
    result = addition_agent.invoke(state)
    print("==========ADDITION AGENT RESULT==========")
    print(result)
    goto = get_next_node(result["messages"][-1], MULTIPLICATION_AGENT)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name=ADDITION_AGENT
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto,
    )

    
def multiplication_agent_function(state)  -> Command[Literal[ADDITION_AGENT, END]]:
    print("==========MULTIPLICATION AGENT==========")
    print(state)
    result = multiplication_agent.invoke(state)
    print("==========MULTIPLICATION AGENT RESULT==========")
    print(result)
    goto = get_next_node(result["messages"][-1], ADDITION_AGENT)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name=MULTIPLICATION_AGENT
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto,
    )

def multiply(a, b):
    """Multiply the two numbers.  You can't perfrom any other operations.\n\nIf you have completed all tasks, respond with FINAL ANSWER."""
    return (a * b)

def add(a, b):
    """Add the two numbers.  You can't perfrom any other operations.\n\nIf you have completed all tasks, respond with FINAL ANSWER."""
    return (a + b)


def get_next_node(last_message, goto):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

embeddings = OpenAIEmbeddings()
tools = [multiply, add]
llm = ChatOpenAI(model_name="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# Research agent and node
addition_agent = create_react_agent(
    llm,
    tools=[add],
    prompt=make_system_prompt(
        "You can only do addition. You are working with a multiplication colleague. You can't perform any other operations."
    ),
)

multiplication_agent = create_react_agent(
    llm,
    tools=[multiply],
    prompt=make_system_prompt(
        "You can only do multiplication. You are working with an addition colleague. You can't perform any other operations."
    ),
)


