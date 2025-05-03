from common_imports import *
from constants import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-network-add-multiply-with-chain-2.png")

@tool
def multiply(a, b):
    """Multiply the two numbers.  You can't perfrom any other operations."""
    return (a * b)

@tool
def add(a, b):
    """Add the two numbers.  You can't perfrom any other operations."""
    return (a + b)


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants.\n"
        " Use the provided tools to progress towards answering the question.\n"
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress.\n"
        " If you or any of the other assistants have the final answer or deliverable,\n"
        " prefix your response with FINAL ANSWER so the team knows to stop.\n"
        f"\n{suffix}"
    )

def addition_agent_function(state):
    print("==========ADDITION AGENT==========")
    print(state)
    question = state["question"]
    context = state["messages"]
    system_prompt = make_system_prompt(suffix="""You are an addition agent.\n
                                       You will be given two numbers and you will need to add them together using the tool given to you.\n
                                       You will be working with a multiplication agent too.\n
                                       Don't perform any other operations except addition.""")
    addition_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question} \n"),
    ])
    llm_with_tools = llm.bind_tools([add])
    addition_chain = addition_prompt | llm_with_tools
    invoke_message = {"question": question}
    result = addition_chain.invoke(invoke_message)
    print("==========ADDITION AGENT RESULT==========")
    print(result)
    return {"messages": [result]}

    
def multiplication_agent_function(state):
    print("==========MULTIPLICATION AGENT==========")
    print(state)
    question = state["question"]
    context = state["messages"]
    system_prompt = make_system_prompt(suffix="""You are a multiplication agent.\n
                                       You will be given two numbers and you will need to multiply them together using the tool given to you.\n
                                       You will be working with an addition agent too.\n
                                       Don't perform any other operations except multiplication.""")
    multiplication_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question} \n"),
    ])
    llm_with_tools = llm.bind_tools([multiply])
    multiplication_chain = multiplication_prompt | llm_with_tools
    invoke_message = {"question": question}
    result = multiplication_chain.invoke(invoke_message)
    print("==========MULTIPLICATION AGENT RESULT==========")
    print(result)
    return {"messages": [result]}

def addition_agent_router(state):
    print("==========ADDITION AGENT ROUTER==========")
    print(state)
    if "FINAL ANSWER" in state["messages"][-1].content:
        return END
    return MULTIPLICATION_AGENT

def multiplication_agent_router(state):
    print("==========MULTIPLICATION AGENT ROUTER==========")
    print(state)
    if "FINAL ANSWER" in state["messages"][-1].content:
        return END
    return ADDITION_AGENT

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini")