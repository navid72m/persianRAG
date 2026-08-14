from .config import CFG
from .embeddings import embed_query
from .reranker import rerank
from .ingest import SPARSE_VOCAB_PATH
from .parent_store import get_parents
from .sparse import BM25Vocab
from .vectorstore import get_client, hybrid_search

_client = get_client()
_vocab = None


def _vocab_singleton() -> BM25Vocab:
    global _vocab
    if _vocab is None:
        _vocab = BM25Vocab.load(SPARSE_VOCAB_PATH)
    return _vocab


def retrieve_for_query(query: str, top_k: int = None) -> list[dict]:
    top_k = top_k or CFG.retrieve_top_k
    dense_vec = embed_query(query)
    indices, values = _vocab_singleton().encode(query)
    points = hybrid_search(_client, dense_vec, indices, values, top_k)
    return [
        {
            "child_id": p.payload["child_id"],
            "parent_id": p.payload["parent_id"],
            "text": p.payload["text"],
            "page_start": p.payload["page_start"],
            "page_end": p.payload["page_end"],
            "score": p.score,
        }
        for p in points
    ]


def retrieve_and_assemble(queries: list[str]) -> list[dict]:
    """Runs hybrid retrieval for one or more (sub-)queries, dedupes by
    parent_id (keeping the best child score per parent), fetches full
    parent text, and reranks the parents against the *original* combined
    query context. Returns top rerank_top_k parent chunks ready for
    generation.
    """
    all_hits: list[dict] = []
    for q in queries:
        all_hits.extend(retrieve_for_query(q))

    best_per_parent: dict[str, dict] = {}
    for hit in all_hits:
        pid = hit["parent_id"]
        if pid not in best_per_parent or hit["score"] > best_per_parent[pid]["score"]:
            best_per_parent[pid] = hit

    parent_ids = list(best_per_parent.keys())
    parents = get_parents(CFG.parent_db_path, parent_ids)

    candidates = [
        {
            "parent_id": pid,
            "text": parents[pid]["text"],
            "page_start": parents[pid]["page_start"],
            "page_end": parents[pid]["page_end"],
            "child_score": best_per_parent[pid]["score"],
        }
        for pid in parent_ids if pid in parents
    ]

    if not candidates:
        return []

    rerank_query = " | ".join(queries)
    rerank_results = rerank(rerank_query, [c["text"] for c in candidates], top_n=CFG.rerank_top_k)

    ranked = []
    for r in rerank_results:
        c = candidates[r["index"]]
        ranked.append({**c, "rerank_score": r["relevance_score"]})
    return ranked
