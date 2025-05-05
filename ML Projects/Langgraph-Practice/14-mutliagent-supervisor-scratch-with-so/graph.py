from functions import supervisor_agent, research_node, math_node, finance_node
from constants import *
from common_imports import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(SUPERVISOR_AGENT, supervisor_agent)
workflow.add_node(RESEARCH_AGENT, research_node)
workflow.add_node(MATH_AGENT, math_node)
workflow.add_node(FINANCE_AGENT, finance_node)

workflow.add_edge(START, SUPERVISOR_AGENT)

graph = workflow.compile()

