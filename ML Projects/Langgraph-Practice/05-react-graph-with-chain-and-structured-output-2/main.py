from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from React Graph with Chain and Structured Output 2!")
    # query = """"Find the stock price of the most valuable company in the world."""
    # instructions = """Also, give a brief summary of the company in 2-3 sentences."""
    query = "What's the richest city in the world?"
    instructions = "Also, give a brief summary of the place in 2-3 sentences."
    displayGraph(graph) 
    response = graph.invoke({"question": query, "instructions": instructions})
    for msg in response['messages']:
        msg.pretty_print()

    if response.get('final_response', None) is not None:
        print("========STRUCTURED OUTPUT========")
        print(response['final_response'])
    else:
        print("========OUTPUT========")
        print(response['messages'][-1].content)