"""Qdrant collection: children only, with two named vectors (dense + sparse),
fused with Qdrant's built-in RRF at query time.
"""
from qdrant_client import QdrantClient, models

from .config import CFG


def get_client() -> QdrantClient:
    return QdrantClient(url=CFG.qdrant_url, api_key=CFG.qdrant_api_key)


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(CFG.collection_name):
        info = client.get_collection(CFG.collection_name)
        v = info.config.params.vectors
        size = v["dense"].size if isinstance(v, dict) else v.size
        if size == CFG.embed_dim:
            return
        print(f"Collection {CFG.collection_name} has dim {size}, need {CFG.embed_dim} — "
              "recreating (children are rebuildable by re-running ingest).")
        client.delete_collection(CFG.collection_name)
    client.create_collection(
        collection_name=CFG.collection_name,
        vectors_config={
            "dense": models.VectorParams(size=CFG.embed_dim, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def upsert_children(client: QdrantClient, points: list[dict]) -> None:
    """points: [{id, dense_vector, sparse_indices, sparse_values, payload}, ...]"""
    qpoints = [
        models.PointStruct(
            id=p["id"],
            vector={
                "dense": p["dense_vector"],
                "sparse": models.SparseVector(indices=p["sparse_indices"], values=p["sparse_values"]),
            },
            payload=p["payload"],
        )
        for p in points
    ]
    client.upsert(collection_name=CFG.collection_name, points=qpoints, wait=True)


def hybrid_search(client: QdrantClient, dense_vector: list[float],
                   sparse_indices: list[int], sparse_values: list[float],
                   top_k: int) -> list[models.ScoredPoint]:
    result = client.query_points(
        collection_name=CFG.collection_name,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return result.points
