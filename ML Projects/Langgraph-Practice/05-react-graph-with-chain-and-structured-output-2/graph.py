from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(REASONER, reasoner)  # REASONER
workflow.add_node(TOOLS, ToolNode(get_tools()))  # TOOLS
workflow.add_node(RESPONDER, responder)  # RESPONDER
workflow.add_node(OUTPUT_DECIDER, output_decider)  # OUTPUT_DECIDER

workflow.set_entry_point(REASONER)
workflow.add_conditional_edges(
    REASONER,
    reasoner_router,
    {
        TOOLS: TOOLS,
        OUTPUT_DECIDER: OUTPUT_DECIDER,
    },
)
workflow.add_conditional_edges(
    OUTPUT_DECIDER,
    output_decider_router,
    {
        RESPONDER: RESPONDER,
        END: END,
    },
)   
workflow.add_edge(TOOLS, REASONER)
workflow.add_edge(RESPONDER, END)
graph = workflow.compile()