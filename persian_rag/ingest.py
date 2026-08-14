import os
import sys

from .chunking import chunk_document
from .config import CFG
from .embeddings import embed_documents
from .parent_store import save_parents
from .sparse import BM25Vocab
from .vectorstore import ensure_collection, get_client, upsert_children

SPARSE_VOCAB_PATH = os.getenv("SPARSE_VOCAB_PATH", "sparse_vocab.json")
BATCH_SIZE = 96  # Cohere embed batch limit is generous, but keep requests moderate


def main(pdf_path: str, ocr_enabled: bool) -> None:
    print(f"Parsing + chunking {pdf_path} ...")
    parents, children = chunk_document(
        pdf_path,
        parent_tokens=CFG.parent_chunk_tokens,
        child_tokens=CFG.child_chunk_tokens,
        child_overlap=CFG.child_chunk_overlap,
        ocr_enabled=ocr_enabled and CFG.ocr_enabled,
        ocr_lang=CFG.ocr_lang,
        ocr_dpi=CFG.ocr_dpi,
    )
    print(f"  {len(parents)} parent chunks, {len(children)} child chunks")
    if not parents:
        print("WARNING: no text extracted. If the PDF is scanned, re-run WITHOUT --no-ocr.")

    print("Saving parent chunks to sqlite ...")
    save_parents(CFG.parent_db_path, parents)

    print("Fitting BM25 vocabulary over child chunks ...")
    vocab = BM25Vocab()
    vocab.fit([c.text for c in children])
    vocab.save(SPARSE_VOCAB_PATH)

    print("Connecting to Qdrant ...")
    client = get_client()
    ensure_collection(client)

    print("Embedding + upserting children in batches ...")
    for i in range(0, len(children), BATCH_SIZE):
        batch = children[i:i + BATCH_SIZE]
        dense_vectors = embed_documents([c.text for c in batch])
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
        upsert_children(client, points)
        print(f"  upserted {min(i + BATCH_SIZE, len(children))}/{len(children)}")

    print("Done.")


def _stable_int_id(child_id: str) -> int:
    # Qdrant point ids must be int or UUID; derive a stable int from the child_id string.
    import hashlib
    h = hashlib.sha1(child_id.encode()).hexdigest()
    return int(h[:12], 16)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ocr_enabled = "--no-ocr" not in sys.argv
    if len(args) != 1:
        print("Usage: python -m persian_rag.ingest [--no-ocr] /path/to/document.pdf")
        sys.exit(1)
    main(args[0], ocr_enabled=ocr_enabled)
