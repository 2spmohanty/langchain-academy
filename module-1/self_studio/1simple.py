import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import START,END, StateGraph



class State(TypedDict):
    graph_state: str


def toss(state: State) -> Literal["node2","node3"]:

    user_input = state["graph_state"]
    if random.random() < 0.5:
        return "node2"
    else:
        return "node3"

def node1(state: State) :
    print("-------- Node1----------")
    return {"graph_state" : state["graph_state"] + " Result: "}

def node2(state: State) :
    print("-------- Node2----------")
    return {"graph_state" : state["graph_state"] + " And India Wins"}

def node3(state: State) :
    print("-------- Node3----------")
    return {"graph_state" : state["graph_state"] + " And Odisha Wins"}


builder = StateGraph(State)
builder.add_node("node1",node1)
builder.add_node("node2",node2)
builder.add_node("node3",node3)
builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", toss)
builder.add_edge("node2", END)
builder.add_edge("node2", END)

graph = builder.compile()

