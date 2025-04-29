from common_imports import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-network.png")

def addition_agent_function(state):
    print("==========ADDITION AGENT==========")
    print(state)
    question = state["question"]
    system_message = """
        You are an addition expert, you can ask the multiplication expert for help with multiplication. \n
        Always do your portion of calculation before the handoff. \n
        You will be given a question by the user.
        """
    addition_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Question: {question}")
    ])
    addition_chain = addition_prompt | llm_with_tools
    invoke_message = {"question": question}
    print("==========ADDITION AGENT INVOKE MESSAGE==========")
    print(invoke_message)
    response = addition_chain.invoke(invoke_message)
    print("==========ADDITION AGENT RESPONSE==========")
    print(response)
    if response.tool_calls:
        tool_call_id = response.tool_calls[0]['id']
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto=MULTIPLICATION_AGENT, update={"messages": [response, tool_msg]})
    return {"messages": [response]}

def multiplication_agent_function(state):
    print("==========MULTIPLICATION AGENT==========")
    print(state)
    question = state["question"]
    system_message = """
        You are a multiplication expert, you can ask the addition expert for help with addition. \n
        Always do your portion of calculation before the handoff. \n
        You will be given a question by the user.
        """
    multiplication_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Question: {question}")
    ])
    multiplication_chain = multiplication_prompt | llm_with_tools
    invoke_message = {"question": question}
    print("==========MULTIPLICATION AGENT INVOKE MESSAGE==========")
    print(invoke_message)
    response = multiplication_chain.invoke(invoke_message)
    print("==========MULTIPLICATION AGENT RESPONSE==========")
    print(response)
    if response.tool_calls:
        tool_call_id = response.tool_calls[0]['id']
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto=ADDITION_AGENT, update={"messages": [response, tool_msg]})
    return {"messages": [response]}

def transfer_to_multiplication_expert():
    """Ask multiplication agent for help"""
    return

def transfer_to_addition_expert():
    """Ask addition agent for help"""
    return

embeddings = OpenAIEmbeddings()
tools = [transfer_to_multiplication_expert, transfer_to_addition_expert]
llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)