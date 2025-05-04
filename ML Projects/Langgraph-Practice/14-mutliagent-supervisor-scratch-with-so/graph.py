from functions import supervisor_agent, research_agent, math_agent, finance_agent
from constants import *
from common_imports import *
from classes import AgentState

graph = (
    StateGraph(AgentState)
    .add_node(supervisor_agent, destinations=(RESEARCH_AGENT, MATH_AGENT, FINANCE_AGENT, END))
    .add_node(research_agent)
    .add_node(math_agent)
    .add_node(finance_agent)
    .add_edge(START, SUPERVISOR_AGENT)
    .add_edge(RESEARCH_AGENT, SUPERVISOR_AGENT)
    .add_edge(MATH_AGENT, SUPERVISOR_AGENT)
    .add_edge(FINANCE_AGENT, SUPERVISOR_AGENT)
    .compile()
)