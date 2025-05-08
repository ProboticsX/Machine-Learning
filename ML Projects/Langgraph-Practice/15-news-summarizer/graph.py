from functions import news_supervisor_agent, top_headlines_node, top_headlines_supervisor_agent, top_headlines_summarizer_node
from constants import *
from common_imports import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(NEWS_SUPERVISOR_AGENT, news_supervisor_agent)
workflow.add_node(TOP_HEADLINES_SUPERVISOR_AGENT, top_headlines_supervisor_agent)
workflow.add_node(TOP_HEADLINES_SUMMARIZER_AGENT, top_headlines_summarizer_node)
workflow.add_node(TOP_HEADLINES_AGENT, top_headlines_node)

workflow.add_edge(START, NEWS_SUPERVISOR_AGENT)

graph = workflow.compile()

