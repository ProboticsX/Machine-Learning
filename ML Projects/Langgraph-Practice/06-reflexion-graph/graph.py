from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(DRAFT, draft)  # DRAFT
workflow.add_node(EXECUTE_TOOLS, execute_tools)  # EXECUTE_TOOLS
workflow.add_node(REVISOR, revisor)  # REVISOR

workflow.set_entry_point(DRAFT)
workflow.add_edge(DRAFT, EXECUTE_TOOLS)
workflow.add_edge(EXECUTE_TOOLS, REVISOR)
workflow.add_conditional_edges(
    REVISOR,
    revisor_router,
    {
        EXECUTE_TOOLS: EXECUTE_TOOLS,
        END: END,
    },
)
graph = workflow.compile()