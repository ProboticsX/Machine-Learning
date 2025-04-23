from common_imports import *
from chains import get_generate_chain, get_reflect_chain

def displayGraph(graph):
    graph.get_graph().draw_mermaid_png(output_file_path="reflection_graph.png")

def generate(state):
    print("===GENERATE===")
    generate_chain, user_tweet = get_generate_chain(state)
    response = generate_chain.invoke({"user_tweet": user_tweet})
    return {"messages": [response]}

def reflect(state):
    print("===REFLECT===")
    reflect_chain, last_tweet = get_reflect_chain(state)
    response = reflect_chain.invoke({"last_tweet": last_tweet})
    return {"messages": [response]}

def generate_router(state):
    if len(state["messages"]) > 6:
        return END
    return "reflect"

