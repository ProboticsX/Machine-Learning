from common_imports import *
from constants import *
from classes import AgentState

def displayGraph(graph):
    # print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="news-summarizer.png")

@tool
def multiply(a, b):
    """Multiply the two numbers"""
    return (a * b)

@tool
def add(a, b):
    """Add the two numbers"""
    return (a + b)

@tool
def divide(a, b):
    """Divide the two numbers"""
    return (a / b)

@tool
def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.
    Args:
        ticker: ticker str
    """
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']

@tool
def get_top_headlines() -> str:
    """Gets the top headlines from the news API."""
    news_tool = NewsContentTool()
    result = news_tool.get_top_headlines()
    return result

def make_news_supervisor_node(llm, members) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def news_supervisor_node(state: AgentState) -> Command[Literal[*members, END]]:
        print("====NEWS SUPERVISOR NODE=====")
        print("===STATE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        print("===INVOKE MESSAGE=====")
        print(invoke_message)
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return news_supervisor_node


def make_top_headlines_supervisor_node(llm, members) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor who is expected to provide the top news headlines along with the summary of the news."
        "You are tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def top_headlines_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWS_SUPERVISOR_AGENT]]:
        print("====TOP HEADLINES SUPERVISOR NODE=====")
        print("===STATE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        print("===INVOKE MESSAGE=====")
        print(invoke_message)
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = NEWS_SUPERVISOR_AGENT
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return top_headlines_supervisor_node

def top_headlines_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES NODE=====")
    print("===STATE=====")
    print(state)
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = "Your job is to provide the top headlines using the tools provided.")
    result = top_headlines_agent.invoke(state)
    print("===RESULT=====")
    print(result)
    tool_message = result["messages"][-2]
    return Command(
        update={
            "messages": [
                HumanMessage(content=tool_message.content, name=TOP_HEADLINES_AGENT)
            ]
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )

def top_headlines_summarizer_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES SUMMARIZER NODE=====")
    print("===STATE=====")
    print(state)
    previous_messages = state["messages"]
    system_prompt = "Your job is to summarize the top headlines. Remember to summarize each headline in about 200-300 words."
    invoke_message = [{"role": "system", "content": system_prompt},] + previous_messages
    result = llm.invoke(invoke_message)
    print("===RESULT=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result.content, name=TOP_HEADLINES_SUMMARIZER_AGENT)
            ]
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )

top_headlines_agent_tools = [get_top_headlines]

llm = ChatOpenAI(model_name="gpt-4o-mini")

news_supervisor_members = [TOP_HEADLINES_SUPERVISOR_AGENT]
news_supervisor_agent = make_news_supervisor_node(llm, members=news_supervisor_members)
top_headlines_members = [TOP_HEADLINES_AGENT, TOP_HEADLINES_SUMMARIZER_AGENT]
top_headlines_supervisor_agent = make_top_headlines_supervisor_node(llm, members=top_headlines_members)