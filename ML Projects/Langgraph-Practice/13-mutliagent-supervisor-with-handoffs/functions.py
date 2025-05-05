from common_imports import *
from constants import *

def displayGraph(graph):
    # print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="multiagent-supervisor-with-handoffs.png")

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

def create_handoff_tool(agent_name, description):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(state: Annotated[MessagesState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        print("\n=== Current State ===")
        print(state)
        print(f"Tool Call ID: {tool_call_id}")
        print(f"Transferring to: {agent_name}")
        print("===================\n")
        
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]}, # update the state with the tool message
            graph=Command.PARENT, # transfer control back to the parent graph (supervisor agent)
        )

    return handoff_tool


embeddings = OpenAIEmbeddings()
math_agent_tools = [multiply, add, divide]
llm = ChatOpenAI(model_name="gpt-4o")

web_search_tool = TavilySearchResults(max_results=3)
research_agent = create_react_agent(
    model=llm,
    tools=[web_search_tool],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name=RESEARCH_AGENT,
)

math_agent = create_react_agent(
    model=llm,
    tools=math_agent_tools,
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name=MATH_AGENT,
)

# Handoffs
assign_to_research_agent = create_handoff_tool(
    agent_name=RESEARCH_AGENT,
    description="Assign task to a researcher agent.",
)

assign_to_math_agent = create_handoff_tool(
    agent_name=MATH_AGENT,
    description="Assign task to a math agent.",
)

supervisor_agent = create_react_agent(
    model=llm,
    tools=[assign_to_research_agent, assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name=SUPERVISOR_AGENT,
)