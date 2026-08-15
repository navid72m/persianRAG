"""Resume an interrupted ingest: re-embed children into Qdrant without
re-running OCR/chunking. Children are deterministically rebuilt from
parents.sqlite (same ids), then embedded + upserted.

Resumable: after each batch a progress marker is written to
EMBED_PROGRESS (default "resume_embed.progress"), so an interrupted run
continues from the last completed batch instead of restarting.

Usage:
    python3 -m persian_rag.resume_embed
    EMBED_BATCH_SIZE=32 python3 -m persian_rag.resume_embed   # lower memory
"""
import os
import sqlite3
import time

from .chunking import ParentChunk, build_child_chunks
from .config import CFG
from .embeddings import embed_documents
from .ingest import SPARSE_VOCAB_PATH, _stable_int_id
from .sparse import BM25Vocab
from .vectorstore import ensure_collection, get_client, upsert_children

BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))  # lower (e.g. 32) if the box OOMs
PROGRESS_PATH = os.getenv("EMBED_PROGRESS", "resume_embed.progress")


def _embed_patiently(texts: list[str], label: str) -> list[list[float]]:
    """Embed one batch, retrying forever on transient failures (the API
    geo-block is intermittent on some networks; local backend never fails)."""
    while True:
        try:
            return embed_documents(texts)
        except RuntimeError as e:
            print(f"  {label}: {e} — waiting 120s and retrying ...", flush=True)
            time.sleep(120)


def _upsert_patiently(client, points: list[dict], label: str) -> None:
    """Upsert one batch, retrying forever on transport errors (upserts are
    idempotent — same stable ids, same content)."""
    while True:
        try:
            upsert_children(client, points)
            return
        except Exception as e:
            print(f"  {label}: upsert failed ({e}) — waiting 60s and retrying ...",
                  flush=True)
            time.sleep(60)


def _read_progress() -> int:
    if os.path.exists(PROGRESS_PATH):
        try:
            n = int(open(PROGRESS_PATH).read().strip())
            return (n // BATCH_SIZE) * BATCH_SIZE  # align to batch boundary
        except (ValueError, OSError):
            pass
    return 0


def _write_progress(n: int) -> None:
    with open(PROGRESS_PATH, "w") as f:
        f.write(str(n))


def main() -> None:
    print("Loading parents from sqlite ...", flush=True)
    conn = sqlite3.connect(CFG.parent_db_path)
    rows = conn.execute("SELECT parent_id, text, page_start, page_end FROM parents").fetchall()
    conn.close()
    parents = [ParentChunk(r[0], r[1], r[2], r[3]) for r in rows]
    print(f"  {len(parents)} parents", flush=True)

    children = build_child_chunks(parents, CFG.child_chunk_tokens, CFG.child_chunk_overlap)
    print(f"  rebuilt {len(children)} children", flush=True)

    start = _read_progress()
    if start:
        print(f"  resuming from batch {start // BATCH_SIZE + 1} ({start}/{len(children)})",
              flush=True)

    print("Loading BM25 vocabulary ...", flush=True)
    vocab = BM25Vocab.load(SPARSE_VOCAB_PATH)
    print(f"  {vocab.n_docs} docs, {len(vocab.term_to_idx)} terms", flush=True)

    print("Connecting to Qdrant ...", flush=True)
    client = get_client()
    ensure_collection(client)

    print("Embedding + upserting children ...", flush=True)
    for i in range(start, len(children), BATCH_SIZE):
        batch = children[i:i + BATCH_SIZE]
        dense_vectors = _embed_patiently([c.text for c in batch], f"batch {i // BATCH_SIZE + 1}")
        points = []
        for child, dvec in zip(batch, dense_vectors):
            indices, values = vocab.encode(child.text)
            points.append({
                "id": _stable_int_id(child.child_id),
                "dense_vector": dvec,
                "sparse_indices": indices,
                "sparse_values": values,
                "payload": {
                    "child_id": child.child_id,
                    "parent_id": child.parent_id,
                    "text": child.text,
                    "page_start": child.page_start,
                    "page_end": child.page_end,
                },
            })
        _upsert_patiently(client, points, f"batch {i // BATCH_SIZE + 1}")
        _write_progress(i + len(batch))
        print(f"  upserted {min(i + BATCH_SIZE, len(children))}/{len(children)}", flush=True)

    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
