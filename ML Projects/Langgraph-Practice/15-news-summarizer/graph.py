from functions.news_summarizer_team import *
from functions.top_headlines_team import *
from functions.podcast_team import *
from common_imports import *
from classes import AgentState

workflow = StateGraph(AgentState)

workflow.add_node(NEWS_SUPERVISOR_AGENT, news_supervisor_agent)

workflow.add_node(TOP_HEADLINES_SUPERVISOR_AGENT, top_headlines_supervisor_agent)
workflow.add_node(TOP_HEADLINES_SUMMARIZER_AGENT, top_headlines_summarizer_node)
workflow.add_node(TOP_HEADLINES_AGENT, top_headlines_node)

workflow.add_node(PODCAST_SUPERVISOR_AGENT, podcast_supervisor_agent)
workflow.add_node(PODCAST_TRANSCRIPT_SUPERVISOR_AGENT, podcast_transcript_supervisor_agent)
workflow.add_node(PODCAST_AUDIO_GENERATOR_AGENT, podcast_audio_generator_node)

workflow.add_node(PODCAST_TRANSCRIPT_WRITER_AGENT, podcast_transcript_writer_node)
workflow.add_node(PODCAST_TRANSCRIPT_GENERATOR_AGENT, podcast_transcript_generator_node)
workflow.add_node(PODCAST_TRANSCRIPT_FINETUNER_AGENT, podcast_transcript_finetuner_node)

workflow.add_edge(START, NEWS_SUPERVISOR_AGENT)

graph = workflow.compile()

