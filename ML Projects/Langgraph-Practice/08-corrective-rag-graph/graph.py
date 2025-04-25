from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(RETRIEVER, retriever_function)  # RETRIEVER
workflow.add_node(GRADER, grader_function)  # GRADER
workflow.add_node(WEBSEARCH, websearch_function)  # WEBSEARCH
workflow.add_node(GENERATOR, generator_function)  # GENERATOR

workflow.set_entry_point(RETRIEVER)
workflow.add_edge(RETRIEVER, GRADER)
workflow.add_conditional_edges(
    GRADER,
    grader_router,
    {
        GENERATOR: GENERATOR,
        WEBSEARCH: WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATOR)
workflow.add_edge(GENERATOR, END)
graph = workflow.compile()