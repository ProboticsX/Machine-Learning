from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(REASONER, reasoner)  # REASONER
workflow.add_node(TOOLS, ToolNode(get_tools()))  # TOOLS
workflow.add_node(RESPONDER, responder)  # RESPONDER
workflow.add_node(OUTPUT_ROUTER, output_router)  # OUTPUT_ROUTER

workflow.set_entry_point(REASONER)
workflow.add_conditional_edges(
    REASONER,
    reasoner_router,
    {
        TOOLS: TOOLS,
        OUTPUT_ROUTER: OUTPUT_ROUTER,
    },
)
workflow.add_conditional_edges(
    OUTPUT_ROUTER,
    output_router,
    {
        RESPONDER: RESPONDER,
        END: END,
    },
)
workflow.add_edge(TOOLS, REASONER)
workflow.add_edge(RESPONDER, END)
graph = workflow.compile()