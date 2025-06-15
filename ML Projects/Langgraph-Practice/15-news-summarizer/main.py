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
    print("Hello from Newspresso!")
    # question = "find US and New York state GDP in 2024. what percentage of US GDP was New York state?? Multiply the result by 369."
    # processsed_news_file_path = NewsTool().processed_news_path
    # print(processsed_news_file_path)
    # question = "What are the latest top headlines in the general category in the last 24 hours? Create a podcast on the same."
    question = "What are the latest top headlines in the entertainment category in the last 24 hours? Create a podcast on the same."
    # question = "RAG: What's the latest news on the US-China trade war?"
    # question = "RAG: What's the latest news on Owen Wilson?"
    # question = "RAG: Tell me about the recent news on Taylor Swift's new album?"
    displayGraph(graph)
    result = graph.invoke({"messages": [question]}, config={"recursion_limit": 35})
    print("======RESULT=======")
    print(result)
    print("======FINAL RESPONSE=======")
    for msg in result['messages']:
        msg.pretty_print()
    print("======RESULT STRUCTURE=======")
    print_dict_structure(result)