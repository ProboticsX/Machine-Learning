from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(AGENT, agent)  # AGENT
workflow.add_node(LLM, llm_function)  # LLM
workflow.add_node(RAG, rag)  # RAG

workflow.set_entry_point(AGENT)
workflow.add_conditional_edges(
    AGENT,
    agent_router,
    {
        LLM: LLM,
        RAG: RAG,
    },
)
workflow.add_edge(LLM, END)
workflow.add_edge(RAG, END)
graph = workflow.compile()