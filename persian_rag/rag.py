import sys

from .graph import build_graph

_graph = None


def _graph_singleton():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, chat_history: list[dict] | None = None) -> dict:
    """Runs one query through the full pipeline. Returns the final state
    (includes route taken, retrieved chunks, and the answer)."""
    graph = _graph_singleton()
    final_state = graph.invoke({"query": query, "chat_history": chat_history or []})
    return final_state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m persian_rag.rag \"پرسش شما\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    state = run_query(q)
    print("\n--- route:", state.get("route"), "| intent:", state.get("intent"), "---\n")
    print(state.get("answer"))
