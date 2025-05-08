from common_imports import *
from constants import *
from classes import AgentState

def displayGraph(graph):
    # print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-supervisor-scratch-with-so.png")

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

def make_supervisor_node(llm, members) -> str:
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

    def supervisor_node(state: AgentState) -> Command[Literal[*members, END]]:
        print("====SUPERVISOR NODE=====")
        print("===STATE=====")
        print(state)
        """An LLM-based router."""
        previous_messages = state["messages"]
        invoke_message = [{"role": "system", "content": system_prompt},] + previous_messages
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

    return supervisor_node


def research_node(state: AgentState) -> Command[Literal[SUPERVISOR_AGENT]]:
    print("====RESEARCH NODE=====")
    print("===STATE=====")
    print(state)
    research_agent = create_react_agent(llm, tools=research_agent_tools)
    result = research_agent.invoke(state)
    print("===RESULT=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=RESEARCH_AGENT)
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto=SUPERVISOR_AGENT,
    )

def finance_node(state: AgentState) -> Command[Literal[SUPERVISOR_AGENT]]:
    print("====FINANCE NODE=====")
    print("===STATE=====")
    print(state)
    finance_agent = create_react_agent(llm, tools=finance_agent_tools)
    result = finance_agent.invoke(state)
    print("===RESULT=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=FINANCE_AGENT)
            ]
        },
        goto=SUPERVISOR_AGENT,
    )

def math_node(state: AgentState) -> Command[Literal[SUPERVISOR_AGENT]]:
    print("====MATH NODE=====")
    print("===STATE=====")
    print(state)
    math_agent = create_react_agent(llm, tools=math_agent_tools)
    result = math_agent.invoke(state)
    print("===RESULT=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=MATH_AGENT)
            ]
        },
        goto=SUPERVISOR_AGENT,
    )

web_search_tool = TavilySearchResults(max_results=5)
math_agent_tools = [multiply, add, divide]
research_agent_tools = [web_search_tool]
finance_agent_tools = [get_stock_price]

llm = ChatOpenAI(model_name="gpt-4o-mini")

members = [RESEARCH_AGENT, FINANCE_AGENT, MATH_AGENT]
supervisor_agent = make_supervisor_node(llm, members=members)