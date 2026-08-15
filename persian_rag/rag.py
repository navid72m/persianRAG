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


def run_query_stream(query: str, chat_history: list[dict] | None = None,
                     on_update=None) -> dict:
    """Same as run_query, but streams each executed node so callers can show
    live progress. on_update(node_name, partial_state) is invoked as each
    node finishes."""
    graph = _graph_singleton()
    final_state: dict = {"query": query, "chat_history": chat_history or []}
    for update in graph.stream(final_state, stream_mode="updates"):
        for node, partial in update.items():
            final_state.update(partial)
            if on_update:
                on_update(node, partial)
    return final_state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m persian_rag.rag \"پرسش شما\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    state = run_query(q)
    print("\n--- route:", state.get("route"), "| intent:", state.get("intent"), "---\n")
    print(state.get("answer"))
