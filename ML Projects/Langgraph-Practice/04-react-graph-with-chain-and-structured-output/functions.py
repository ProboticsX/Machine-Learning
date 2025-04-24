from common_imports import *
from classes import CityDetails
def displayGraph(graph):
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="react_graph_with_chain_and_so.png")

def reasoner(state):
    print("===REASONER===")
    print(state)
    messages = state["messages"]
    question = state["question"]
    instructions = state["instructions"]
    system_prompt = """You are a helpful assistant that can answer questions and help with tasks. \n
                        You are equipped with tools like\n
                            - tavily_search to search the web for information.\n
                            - get_stock_price to get the stock price. \n
                        You are given a question and you need to answer it using the tools provided (if needed). \n
                        You will be given some context as well. \n
                        Optionally, you will be also provided with some instructions to follow (if provided), then you will need to follow those instructions.\n
                        Please stop reasoning when you have the final answer."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the question: {question} and some context: {context} and some instructions: {instructions}"),
    ])
    reasoning_chain = prompt | llm_with_tools
    invoke_message = {"question": question, "context":messages, "instructions":instructions}
    print("===INVOKE MESSAGE=====")
    print(invoke_message)
    result = reasoning_chain.invoke(invoke_message)
    print("========RESULT========")
    print(result)
    return {"messages": [result]}
    
def responder(state):
    print("===RESPONDER===")
    print(state)
    messages = state["messages"]
    print("===INVOKE MESSAGE=====")
    print(messages)
    result = llm_with_structured_output.invoke(messages)
    print("========RESULT========")
    print(result)
    return {"final_response": result}

def reasoner_router(state):
    if state["messages"][-1].tool_calls:
        return TOOLS
    else:
        return RESPONDER

def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.

    Args:
        ticker: ticker str
    """
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']


def tavily_search(search_query: str):
    """Search the web for the query."""
    search = TavilySearchResults()
    return search.invoke(search_query)

def get_tools():
    return [tavily_search, get_stock_price]

llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_tools = llm.bind_tools(get_tools())
llm_with_structured_output = llm.with_structured_output(CityDetails)