from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from Multi-Agent Network Graph with Addition and Multiplication!")
    question = "What's (3 + 5) * 12?"
    # question = "What does the president say about Ketanji Brown Jackson in the state of the union address?"
    # question = "What has caused the escalation in tensions between India and Pakistan in the last 24 hours?"
    displayGraph(graph) 
    response = graph.invoke({"question": question})
    # print("======SUMMARY=======")
    # print(response['messages'][-1].content)