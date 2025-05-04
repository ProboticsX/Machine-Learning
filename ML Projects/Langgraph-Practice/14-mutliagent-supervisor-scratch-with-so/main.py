from common_imports import *
from functions import displayGraph
from graph import graph

if __name__ == "__main__":
    print("Hello from Multi-Agent Supervisor from scratch with Structured Output!")
    # question = "find US and New York state GDP in 2024. what percentage of US GDP was New York state?? Multiply the result by 369."
    # question = "Who's the richest person in the world? Find what company they own and what's the stock price of the company! Finally, multuply the stock price by 369."
    # question = "What's the stock price of the most valuable company in the world? Give me the name of the CEO of the company too."
    question = "Tell me about the city of New York."
    instructions = "Also, give a brief summary of the place in 2-3 sentences."
    displayGraph(graph)
    result = graph.invoke({"messages": [question], "question": question, "instructions": instructions})
    print("======RESULT=======")
    for msg in result['messages']:
        msg.pretty_print()