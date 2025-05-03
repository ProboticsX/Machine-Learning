from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(ADDITION_AGENT, addition_agent_function)  # ADDITION_AGENT
workflow.add_node(MULTIPLICATION_AGENT, multiplication_agent_function)  # MULTIPLICATION_AGENT
workflow.add_edge(START, ADDITION_AGENT)
workflow.add_conditional_edges(
    ADDITION_AGENT,
    addition_agent_router,
    {
        MULTIPLICATION_AGENT: MULTIPLICATION_AGENT,
        END: END,
    },
)
workflow.add_conditional_edges(
    MULTIPLICATION_AGENT,
    multiplication_agent_router,
    {
        ADDITION_AGENT: ADDITION_AGENT,
        END: END,
    },
)
graph = workflow.compile()