from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from React Graph!")
    query = """"Find the stock price of the most valuable company in the world."""
    displayGraph(graph) 
    response = graph.invoke({"messages":[query]})
    for msg in response['messages']:
        msg.pretty_print()