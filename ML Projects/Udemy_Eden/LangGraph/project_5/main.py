from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph import END, START

class State(TypedDict):
    input: str
    user_feedback: str

def step_1(state: State) -> None:
    print("--STEP 1 --")

def human_feedback(state: State) -> None:
    print("--HUMAN FEEDBACK --")

def step_3(state: State) -> None:
    print("--STEP 3 --")

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    initial_input = {"input": "hello world"}
    thread = {"configurable":{"thread_id":"1"}}
    print("Starting 0")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)
    print("Ending 0")

    print(graph.get_state(thread).next)
    user_input = input("Tell me how you want to update the state:")
    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")
    print("--State after update--")
    print(graph.get_state(thread))

    print("Starting 1")
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
    print("Ending 1")