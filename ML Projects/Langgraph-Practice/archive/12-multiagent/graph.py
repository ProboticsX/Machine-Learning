from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(SUPERVISOR, supervisor_function)  # SUPERVISOR
workflow.add_node(MATH_AGENT, math_agent_function)  # MATH_AGENT
workflow.add_node(MATH_AGENT_TOOLS, ToolNode([add, multiply]))  # MATH_AGENT_TOOLS


workflow.add_edge(START, SUPERVISOR)
workflow.add_conditional_edges(
    SUPERVISOR, 
    supervisor_router,
    {
        MATH_AGENT: MATH_AGENT,
        END: END,
    }
)

workflow.add_conditional_edges(
    MATH_AGENT, 
    math_agent_router,
    {
        MATH_AGENT_TOOLS: MATH_AGENT_TOOLS,
        SUPERVISOR: SUPERVISOR,
    }
)

workflow.add_edge(MATH_AGENT_TOOLS, MATH_AGENT)

graph = workflow.compile()