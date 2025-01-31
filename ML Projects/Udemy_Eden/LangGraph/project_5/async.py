import operator
import time
from typing import TypedDict, Annotated, List, Any, Sequence

from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph

load_dotenv()

class State(TypedDict):
    aggregate: Annotated[List, operator.add]
    which: str

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        time.sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate":[self._value]}

def route_bc_or_cd(state: State)-> Sequence[str]:
    if state["which"] == "bc":
        return ["b","c"]
    return ["c","d"]

intermediates = ["b","c","d"]

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
builder.add_node("e", ReturnNodeValue("I'm E"))

builder.add_edge(START, "a")
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    intermediates #so that it doesn't create an edge from "a" to every node in the graph
)
for node in intermediates:
    builder.add_edge(node, "e")
builder.add_edge("e",END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="async_graph_conditional.png")

if __name__ == "__main__":
    print("Hello Async Graph")
    graph.invoke({"aggregate":[], "which":"bc"},{"configurable":{"thread_id":"foo"}})