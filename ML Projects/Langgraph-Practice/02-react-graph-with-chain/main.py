from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from React Graph with Chain!")
    query = """"Find the stock price of the most valuable company in the world."""
    instructions = """Also, give a brief summary of the company in 2-3 sentences."""
    displayGraph(graph) 
    response = graph.invoke({"question": query})
    for msg in response['messages']:
        msg.pretty_print()