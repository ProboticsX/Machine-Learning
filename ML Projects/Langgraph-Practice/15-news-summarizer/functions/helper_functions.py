from common_imports import *

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    # graph.get_graph().draw_mermaid_png(output_file_path="news-summarizer.png")