from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from Reflexion Graph!")
    # query = """"Find the stock price of the most valuable company in the world."""
    # instructions = """Also, give a brief summary of the company in 2-3 sentences."""
    topic = "Any new startups working in the green energy sector."
    instructions = "The summary should be approximately 250 words."
    displayGraph(graph) 
    response = graph.invoke({"topic": topic, "first_instruction": instructions})
    print("======SUMMARY=======")
    print(response['revisor_details'].summary)
    print("=====REFERENCES=====")
    print(response['revisor_details'].references)