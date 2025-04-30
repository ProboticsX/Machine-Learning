from common_imports import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-network.png")

def addition_agent_function(state):
    print("==========ADDITION AGENT==========")
    print(state)
    question = state["question"]
    messages = state["messages"]
    goto = get_next_node(messages, MULTIPLICATION_AGENT)
    system_message = """
        You are an addition expert, you can ask the multiplication expert for help with multiplication. \n
        Always do your portion of calculation before the handoff. \n
        If you have a final answer to the original question, say FINAL ANSWER followed by the result. \n
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
        return Command(goto=goto, update={"messages": [response, tool_msg]})
    return Command(goto=goto, update={"messages": [response]})

def multiplication_agent_function(state):
    print("==========MULTIPLICATION AGENT==========")
    print(state)
    question = state["question"]
    messages = state["messages"]
    goto = get_next_node(messages, ADDITION_AGENT)
    system_message = """
        You are a multiplication expert, you can ask the addition expert for help with addition. \n
        Always do your portion of calculation before the handoff. \n
        If you have a final answer to the original question, say FINAL ANSWER followed by the result. \n
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
        return Command(goto=goto, update={"messages": [response, tool_msg]})
    return Command(goto=goto, update={"messages": [response]})

def multiply(a, b):
    """Multiply the two numbers"""
    return a * b

def add(a, b):
    """Add the two numbers"""
    return a + b    

def get_next_node(messages, goto):
    if len(messages) > 0 and "FINAL ANSWER" in messages[-1].content:
        # Any agent decided the work is done
        return END
    return goto

embeddings = OpenAIEmbeddings()
tools = [multiply, add]
llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)