from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from React Graph with Chain!")
    query = """"Find the stock price of Apple"""
    displayGraph(graph) 
    response = graph.invoke({"messages":[], "question": query})
    for msg in response['messages']:
        msg.pretty_print()