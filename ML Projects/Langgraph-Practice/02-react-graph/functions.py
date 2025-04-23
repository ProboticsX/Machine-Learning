from common_imports import *

def displayGraph(graph):
    # display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    print(graph.get_graph().draw_ascii())
    # graph.get_graph().draw_mermaid_png(output_file_path="react_graph.png")

def reasoner(state):
    print("===REASONER===")
    print(state)
    messages = state["messages"]
    system_prompt = """You are a helpful assistant that can answer questions and help with tasks. \n
                        You are equipped with tools like\n
                            - DuckDuckGoSearchRun to search the web for information.\n
                        You will be given the user query too.
                        Please stop reasoning when you have the final answer."""
    system_message = SystemMessage(content=system_prompt)
    result = llm_with_tools.invoke([system_message]+messages)
    return {"messages": [result]}
    


def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.

    Args:
        ticker: ticker str
    """
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']


def tavily_search():
    """Search the web for the query."""
    return TavilySearchResults()

def get_tools():
    return [tavily_search, get_stock_price]

llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_tools = llm.bind_tools(get_tools())