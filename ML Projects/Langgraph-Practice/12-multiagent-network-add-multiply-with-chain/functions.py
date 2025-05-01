from common_imports import *
from constants import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-network-add-multiply-with-chain.png")

def addition_agent_function(state) -> Command[Literal[MULTIPLICATION_AGENT, END]]:
    print("==========ADDITION AGENT==========")
    print(state)
    input = state["messages"][0].content
    tool_functions = [add]
    tool_names = ["add"]
    agent_scratchpad = state["messages"]
    custom_prompt = make_system_prompt(
        "You can only do addition. You are working with a multiplication colleague. You can't perform any other operations."
    )
    s = """
    Answer the following questions as best you can. You have access to the tools which will be provided by the user \n\n
    Use the following format:\n\n
        Question: the input question you must answer\n
        Thought: you should always think about what to do\n
        Action: the action to take, should be one of the given tool names provided by the user\n
        Action Input: the input to the action\n
        Observation: the result of the action\n... 
        (this Thought/Action/Action Input/Observation can repeat N times)\n
        Thought: I now know the final answer\n
        Final Answer: the final answer to the original input question\n
        Begin!\n
        Question: will be provided by the user\n
        Thought: will be provided by the user \n

        So you will be given the following input: \n
        - Question: the input question you must answer\n
        - Tool functions: the tool functions you can use\n
        - Tool names: the tool names you can use\n
        - Agent scratchpad: the agent scratchpad you can use for though process
    """
    addition_agent_prompt = ChatPromptTemplate.from_messages(
        [("system", custom_prompt+ s), ("user", 'Question: {input}\n Tool funcitions: {tool_functions}\n Tool names: {tool_names}\n Agent scratchpad: {agent_scratchpad}')]
    )
    addition_agent_chain = addition_agent_prompt | llm
    print("==========ADDITION AGENT INVOKE MESSAGE==========")
    invoke_message = {"input": input, "agent_scratchpad": agent_scratchpad, "tool_names": tool_names, "tool_functions": tool_functions}
    print(invoke_message)
    result = addition_agent_chain.invoke(invoke_message)
    print("==========ADDITION AGENT RESULT==========")
    print(result)
    goto = get_next_node(result, MULTIPLICATION_AGENT)
    return Command(
        update={
            "messages": [result],
        },
        goto=goto,
    )

    
def multiplication_agent_function(state)  -> Command[Literal[ADDITION_AGENT, END]]:
    print("==========MULTIPLICATION AGENT==========")
    print(state)
    input = state["messages"][0].content
    tool_functions = [multiply]
    tool_names = ["multiply"]
    agent_scratchpad = state["messages"]
    custom_prompt = make_system_prompt(
        "You can only do multiplication. You are working with an addition colleague. You can't perform any other operations."
    )
    s = """
    Answer the following questions as best you can. You have access to the tools which will be provided by the user \n\n
    Use the following format:\n\n
        Question: the input question you must answer\n
        Thought: you should always think about what to do\n
        Action: the action to take, should be one of the given tool names provided by the user\n
        Action Input: the input to the action\n
        Observation: the result of the action\n... 
        (this Thought/Action/Action Input/Observation can repeat N times)\n
        Thought: I now know the final answer\n
        Final Answer: the final answer to the original input question\n
        Begin!\n
        Question: will be provided by the user\n
        Thought: will be provided by the user \n

        So you will be given the following input: \n
        - Question: the input question you must answer\n
        - Tool functions: the tool functions you can use\n
        - Tool names: the tool names you can use\n
        - Agent scratchpad: the agent scratchpad you can use for though process
    """
    multiplication_agent_prompt = ChatPromptTemplate.from_messages(
        [("system", custom_prompt+ s), ("user", 'Question: {input}\n Tool funcitions: {tool_functions}\n Tool names: {tool_names}\n Agent scratchpad: {agent_scratchpad}')]
    )
    multiplication_agent_chain = multiplication_agent_prompt | llm
    print("==========MULTIPLICATION AGENT INVOKE MESSAGE==========")
    invoke_message = {"input": input, "agent_scratchpad": agent_scratchpad, "tool_names": tool_names, "tool_functions": tool_functions}
    print(invoke_message)
    result = multiplication_agent_chain.invoke(invoke_message)
    print("==========MULTIPLICATION AGENT RESULT==========")
    print(result)
    goto = get_next_node(result, ADDITION_AGENT)

    return Command(
        update={
            "messages": [result],
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
llm = ChatOpenAI(model_name="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")

# Research agent and node
# addition_agent = create_react_agent(
#     llm,
#     tools=[add],
#     prompt=make_system_prompt(
#         "You can only do addition. You are working with a multiplication colleague. You can't perform any other operations."
#     ),
# )

# multiplication_agent = create_react_agent(
#     llm,
#     tools=[multiply],
#     prompt=make_system_prompt(
#         "You can only do multiplication. You are working with an addition colleague. You can't perform any other operations."
#     ),
# )


