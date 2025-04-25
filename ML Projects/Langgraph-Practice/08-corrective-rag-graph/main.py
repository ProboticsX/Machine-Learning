from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from Corrective RAG Graph!")
    question = "What's the latest news on the conflict between India and Pakistan? Give me a detailed report explaining the full story in how many words you want. How many people have died in the conflict?"
    displayGraph(graph) 
    response = graph.invoke({"question": question})
    print("======SUMMARY=======")
    print(response['messages'][-1].content)