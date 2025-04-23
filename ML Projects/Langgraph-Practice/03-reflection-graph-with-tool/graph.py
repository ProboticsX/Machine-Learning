from common_imports import *
from functions import draft, execute_tools, revise, revisor_router

workflow = StateGraph(MessagesState)

workflow.add_node(DRAFT, draft)  # DRAFT
workflow.add_node(EXECUTE_TOOLS, execute_tools)  # EXECUTE_TOOLS
workflow.add_node(REVISOR, revise)  # REVISOR

workflow.set_entry_point(DRAFT)
workflow.add_edge(DRAFT, EXECUTE_TOOLS)
workflow.add_edge(EXECUTE_TOOLS, REVISOR)

workflow.add_conditional_edges(
    REVISOR,
    revisor_router,
    {
        EXECUTE_TOOLS: EXECUTE_TOOLS,
        END: END
    },
)

graph = workflow.compile()