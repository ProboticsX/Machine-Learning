from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from Multi-Agent Network Graph with Addition and Multiplication with Chain!")
    question = "What's (3 + (5*6)) * 12?"
    displayGraph(graph)
    result = graph.invoke({"messages": [("user", question)]})
    print("======RESULT=======")
    for msg in result['messages']:
        msg.pretty_print()