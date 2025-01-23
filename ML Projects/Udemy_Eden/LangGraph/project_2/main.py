from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import MessageGraph, END

from tool_executor import tool_node
from chains import first_responder, revisor

load_dotenv()
MAX_ITERATIONS = 2

DRAFT = "draft"
EXECUTE_TOOLS = "execute_tools"
REVISOR = "revisor"

builder = MessageGraph()
builder.add_node(DRAFT, first_responder)
builder.add_node(EXECUTE_TOOLS, tool_node)
builder.add_node(REVISOR, revisor)

builder.add_edge(DRAFT, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISOR)

def event_loop(state: List[BaseMessage])->str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations>MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS

builder.add_conditional_edges(REVISOR, event_loop)
builder.set_entry_point(DRAFT)
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflexion_graph.png")

if __name__=="__main__":
    print("Hello from Reflexion Agent")
    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )
    print("SUMMARY:\n")
    print(res[-1].tool_calls[0]["args"]["answer"])
    print("REFERENCES:\n")
    print(res[-1].tool_calls[0]["args"]["references"])
    print("res object:\n")
    print(res)