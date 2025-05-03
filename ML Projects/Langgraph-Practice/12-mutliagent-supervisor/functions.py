from common_imports import *
from constants import *

def multiply(a, b):
    """Multiply the two numbers"""
    return (a * b)

def add(a, b):
    """Add the two numbers"""
    return (a + b)

def divide(a, b):
    """Divide the two numbers"""
    return (a / b)


embeddings = OpenAIEmbeddings()
math_agent_tools = [multiply, add, divide]
llm = ChatOpenAI(model_name="gpt-4o-mini")

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
    name="research_agent",
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
    name="math_agent",
)

supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

supervisor.get_graph().draw_mermaid_png(output_file_path="multiagent-supervisor.png")
question = "find US and New York state GDP in 2024. what percentage of US GDP was New York state?? Multiply the GDP of US by 369."
result = supervisor.invoke({"messages": [("user", question)]})
print(result)

print("==========RESULT==========")
print(result["messages"][-1].content)