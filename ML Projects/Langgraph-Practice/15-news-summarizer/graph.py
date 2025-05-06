from functions import news_supervisor_node, finance_news_node
from constants import *
from common_imports import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(NEWS_SUPERVISOR_AGENT, news_supervisor_node)
workflow.add_node(FINANCE_NEWS_AGENT, finance_news_node)

workflow.add_edge(START, NEWS_SUPERVISOR_AGENT)

graph = workflow.compile()

