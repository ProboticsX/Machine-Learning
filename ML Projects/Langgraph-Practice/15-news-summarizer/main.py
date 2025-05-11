from common_imports import *
from functions.helper_functions import displayGraph
from graph import graph

if __name__ == "__main__":
    print("Hello from News Summarizer!")
    # question = "find US and New York state GDP in 2024. what percentage of US GDP was New York state?? Multiply the result by 369."
    # question = "Who's the richest person in the world? Find what company they own and what's the stock price of the company! Finally, multuply the stock price by 369."
    # question = "What's the stock price of the most valuable company in the world? Give me the name of the CEO of the company too."
    # question = "What's the stock price of Tesla?"
    # question = "Give me some information about Najafgarh"
    # question = "What is ((3*5) + 10)*5? Also, who's the richest person in the world?"
    question = "Create a podcast for the top headlines of the day."
    displayGraph(graph)
    result = graph.invoke({"messages": [question]})
    print("======RESULT=======")
    print(result)
    print("======FINAL RESPONSE=======")
    for msg in result['messages']:
        msg.pretty_print()