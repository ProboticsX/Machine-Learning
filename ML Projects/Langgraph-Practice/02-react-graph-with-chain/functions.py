from common_imports import *

def displayGraph(graph):
    # display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    print(graph.get_graph().draw_ascii())
    # graph.get_graph().draw_mermaid_png(output_file_path="react_graph.png")

def reasoner(state):
    print("===REASONER===")
    print(state)
    messages = state["messages"]
    question = state["question"]
    system_msg = """You are a helpful assistant that can answer questions and help with tasks. \n
                        You are equipped with tools like\n
                            - DuckDuckGoSearchRun to search the web for information.\n
                            - Yahoo Finance tool to get the stock price. \n
                        You are given a question and you need to answer it using the tools provided (if needed). \n
                        Please stop reasoning when you have the final answer."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "Here is the question: {question}"),
    ])
    reasoning_chain = prompt | llm_with_tools
    result = reasoning_chain.invoke({"question": question})
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