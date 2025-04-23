from common_imports import *
from functions import *

workflow = StateGraph(MessagesState)

workflow.add_node(REASONER, reasoner)  # REASONER
workflow.add_node(TOOLS, ToolNode(get_tools()))  # TOOLS

workflow.set_entry_point(REASONER)
workflow.add_conditional_edges(
    REASONER,
    tools_condition,
)
workflow.add_edge(TOOLS, REASONER)
graph = workflow.compile()