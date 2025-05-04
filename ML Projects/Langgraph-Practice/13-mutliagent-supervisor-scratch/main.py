from common_imports import *
from functions import displayGraph
from graph import graph

if __name__ == "__main__":
    print("Hello from Multi-Agent Supervisor from scratch!")
    question = "find US and New York state GDP in 2024. what percentage of US GDP was New York state?? Multiply the result by 369."
    displayGraph(graph)
    result = graph.invoke({"messages": [("user", question)]})
    print("======RESULT=======")
    for msg in result['messages']:
        msg.pretty_print()