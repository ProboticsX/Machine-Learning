from common_imports import *
from constants import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-network.png")

def supervisor_function(state):
    print("==========SUPERVISOR==========")
    print(state)
    context = state["messages"]
    system_prompt = """
    You are a helpful AI assistant, supervising the work of other agents. \n
    You are responsible for answering the user's question. \n
    You can delegate the work to other agents if you need to. \n
    Do not do any work which other agents are able to do yourself. \n
    If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop. \n
    You have access to the following agents: \n
    - Math Agent: Can do math calculations. \n
    """
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the context: {context}")
    ])
    invoke_messages = {
        "context": context
    }
    supervisor_agent_chain = supervisor_prompt | llm.bind_tools([math_agent_function])
    response = supervisor_agent_chain.invoke(invoke_messages)
    print("==========SUPERVISOR RESPONSE==========")
    print(response)
    return {"messages": [response]}

@tool
def math_agent_function(state):
    print("==========MATH AGENT==========")
    print(state)
    context = state["messages"]
    system_prompt = """
    You are a helpful AI assistant, who can do math calculations. \n
    You are provided with the tools to perform the calculations. \n
    Prefix your response with FINAL ANSWER if you have the final answer. \n
    """
    math_agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the context: {context}")
    ])
    invoke_messages = {
        "context": context
    }
    math_agent_chain = math_agent_prompt | llm_with_tools
    response = math_agent_chain.invoke(invoke_messages)
    print("==========MATH AGENT RESPONSE==========")
    print(response)
    return {"messages": [response]}

def supervisor_router(state):
    print("==========SUPERVISOR ROUTER==========")
    if state["messages"][-1].tool_calls:
        return MATH_AGENT
    return END

def math_agent_router(state):
    print("==========MATH AGENT ROUTER==========")
    if state["messages"][-1].tool_calls:
        return MATH_AGENT_TOOLS
    return SUPERVISOR

@tool
def multiply(a, b):
    """Multiply the two numbers.  You can't perfrom any other operations."""
    return (a * b)

@tool
def add(a, b):
    """Add the two numbers.  You can't perfrom any other operations."""
    return (a + b)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply, add])