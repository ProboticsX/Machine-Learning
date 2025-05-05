from functions import supervisor_agent, research_agent, math_agent
from constants import *
from common_imports import *

graph = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent, destinations=(RESEARCH_AGENT, MATH_AGENT, END))
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, SUPERVISOR_AGENT)
    .add_edge(RESEARCH_AGENT, SUPERVISOR_AGENT)
    .add_edge(MATH_AGENT, SUPERVISOR_AGENT)
    .compile()
)