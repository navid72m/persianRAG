from typing import TypedDict

from langgraph.graph import END, StateGraph

from .generate import generate_answer, generate_calculated_answer, generate_direct_answer, extract_calculation
from .query_processing import process_query
from .retrieval import retrieve_and_assemble


class RAGState(TypedDict, total=False):
    query: str
    chat_history: list[dict]
    intent: str
    needs_retrieval: bool
    needs_calculation: bool
    rewritten_query: str
    sub_queries: list[str]
    route: str
    retrieved: list[dict]
    answer: str


def node_rewrite_and_classify(state: RAGState) -> RAGState:
    result = process_query(state["query"], state.get("chat_history"))
    return {**state, **result}


def node_retrieve(state: RAGState) -> RAGState:
    queries = state.get("sub_queries") or [state["rewritten_query"]]
    retrieved = retrieve_and_assemble(queries)
    return {**state, "retrieved": retrieved}


def node_generate(state: RAGState) -> RAGState:
    if state.get("needs_calculation"):
        calc = extract_calculation(state["rewritten_query"], state.get("retrieved", []))
        answer = generate_calculated_answer(state["rewritten_query"], calc, state.get("retrieved", []))
    else:
        answer = generate_answer(state["rewritten_query"], state.get("retrieved", []))
    return {**state, "answer": answer}


def node_direct_answer(state: RAGState) -> RAGState:
    answer = generate_direct_answer(state["rewritten_query"], state.get("chat_history"))
    return {**state, "answer": answer}


def _route_selector(state: RAGState) -> str:
    route = state.get("route", "simple_retrieval")
    if route == "direct_answer" or not state.get("needs_retrieval", True):
        return "direct_answer"
    return "retrieve"  # simple_retrieval and multi_hop both go through the same retrieve node


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rewrite_and_classify", node_rewrite_and_classify)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("direct_answer", node_direct_answer)

    graph.set_entry_point("rewrite_and_classify")
    graph.add_conditional_edges(
        "rewrite_and_classify",
        _route_selector,
        {"retrieve": "retrieve", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()
