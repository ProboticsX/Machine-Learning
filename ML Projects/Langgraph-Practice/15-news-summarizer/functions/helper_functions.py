from common_imports import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="news-summarizer.png")

role_of_each_random_worker = {
    TOP_HEADLINES_CRITIC_AGENT: "Agent who is tasked with providing the critique of the top headlines summary.",
}