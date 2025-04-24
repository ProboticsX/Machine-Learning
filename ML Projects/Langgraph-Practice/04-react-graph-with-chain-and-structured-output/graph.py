from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(REASONER, reasoner)  # REASONER
workflow.add_node(TOOLS, ToolNode(get_tools()))  # TOOLS
workflow.add_node(RESPONDER, responder)  # RESPONDER

workflow.set_entry_point(REASONER)
workflow.add_conditional_edges(
    REASONER,
    reasoner_router,
    {
        TOOLS: TOOLS,
        RESPONDER: RESPONDER,
    },
)
workflow.add_edge(TOOLS, REASONER)
workflow.add_edge(RESPONDER, END)
graph = workflow.compile()