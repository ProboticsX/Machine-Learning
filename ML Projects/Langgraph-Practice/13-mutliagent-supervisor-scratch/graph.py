from functions import supervisor_agent, research_agent, math_agent
from constants import *
from common_imports import *

workflow = StateGraph(MessagesState)

workflow.add_node(SUPERVISOR_AGENT,supervisor_agent)
workflow.add_node(RESEARCH_AGENT, research_agent)
workflow.add_node(MATH_AGENT, math_agent)

workflow.set_entry_point(SUPERVISOR_AGENT)
workflow.add_edge(RESEARCH_AGENT, SUPERVISOR_AGENT)
workflow.add_edge(MATH_AGENT, SUPERVISOR_AGENT)

graph = workflow.compile()


