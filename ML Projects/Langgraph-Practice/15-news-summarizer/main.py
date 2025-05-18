from common_imports import *
from functions.helper_functions import displayGraph
from graph import graph

def print_dict_structure(d, indent=0):
    """Recursively print dictionary keys and values with proper indentation."""
    for key, value in d.items():
        print("  " * indent + f"Key: {key}")
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)
        elif isinstance(value, list):
            print("  " * (indent + 1) + "Value: [List with", len(value), "items]")
            for i, item in enumerate(value):
                print("  " * (indent + 2) + f"Item {i}:")
                if isinstance(item, dict):
                    print_dict_structure(item, indent + 3)
                else:
                    print("  " * (indent + 3) + str(item))
        else:
            print("  " * (indent + 1) + f"Value: {value}")

if __name__ == "__main__":
    print("Hello from News Summarizer!")
    # question = "find US and New York state GDP in 2024. what percentage of US GDP was New York state?? Multiply the result by 369."
    # question = "Who's the richest person in the world? Find what company they own and what's the stock price of the company! Finally, multuply the stock price by 369."
    # question = "What's the stock price of the most valuable company in the world? Give me the name of the CEO of the company too."
    # question = "What's the stock price of Tesla?"
    # question = "Give me some information about Najafgarh"
    # question = "What is ((3*5) + 10)*5? Also, who's the richest person in the world?"
    # processsed_news_file_path = NewsTool().processed_news_path
    # print(processsed_news_file_path)
    question = "What are the top headlines in the world of finance for the day? Create a podcast on the same."
    displayGraph(graph)
    result = graph.invoke({"messages": [question]}, config={"recursion_limit": 35})
    print("======RESULT=======")
    print(result)
    print("======FINAL RESPONSE=======")
    for msg in result['messages']:
        msg.pretty_print()
    print("======RESULT STRUCTURE=======")
    print_dict_structure(result)