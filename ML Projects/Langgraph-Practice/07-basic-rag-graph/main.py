from common_imports import *
from graph import graph
from functions import displayGraph


if __name__ == "__main__":
    print("Hello from Basic RAG Graph!")
    question = "What did the president say about Ketanji Brown Jackson?"
    displayGraph(graph) 
    response = graph.invoke({"question": question})
    print("======SUMMARY=======")
    print(response['messages'][-1].content)