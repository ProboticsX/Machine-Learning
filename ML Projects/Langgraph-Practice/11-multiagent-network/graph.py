from common_imports import *
from functions import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(ADDITION_AGENT, addition_agent_function)  # ADDITION_AGENT
workflow.add_node(MULTIPLICATION_AGENT, multiplication_agent_function)  # MULTIPLICATION_AGENT

workflow.add_edge(START, ADDITION_AGENT)
graph = workflow.compile()