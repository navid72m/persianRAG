"""
Jina Embeddings v3 — dual backend.

Local backend (default, recommended for servers/VPS):
  sentence-transformers, model `jinaai/jina-embeddings-v3`, task prompts
  "retrieval.query"/"retrieval.passage" (identical to the API's task field).

API backend:
  POST https://api.jina.ai/v1/embeddings (geo-blocked from some networks).

Vectors from both backends are produced by the same model, so Qdrant
collections stay consistent across them.
"""
import time

import requests

from .config import CFG

_BATCH_SIZE = 32  # conservative; the API doesn't document a hard limit

_RETRYABLE = (
    requests.ConnectionError,
    requests.Timeout,
)

_model = None


def _local_model():
    """Lazy-load the local SentenceTransformer (weights download on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = CFG.embed_device
        if device == "cuda" and not torch.cuda.is_available():
            print("  CUDA requested but not available — falling back to CPU")
            device = "cpu"
        _model = SentenceTransformer(CFG.embed_model, trust_remote_code=True, device=device)
    return _model


def _embed_local(texts: list[str], task: str) -> list[list[float]]:
    model = _local_model()
    kwargs = {"batch_size": _BATCH_SIZE, "normalize_embeddings": True}
    if task in (model.prompts or {}):
        kwargs["prompt_name"] = task
    vecs = model.encode(texts, **kwargs)
    return [v.tolist() for v in vecs]


def _retry_until_success(attempt_fn, max_retries: int, backoff: float, batch_label: str):
    """Runs attempt_fn, retrying transient failures (connection errors,
    timeouts, 429/451/5xx) with exponential backoff. 451 is Jina's geo-block,
    which is intermittent on some networks — retrying rides it out."""
    delay = backoff
    status = "network error"
    for attempt in range(max_retries + 1):
        try:
            resp = attempt_fn()
            if resp.status_code < 400:
                return resp
            transient = resp.status_code in (408, 429, 451) or resp.status_code >= 500
            if not transient:
                raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text[:300]}")
            status = f"HTTP {resp.status_code}"
        except _RETRYABLE:
            pass
        if attempt == max_retries:
            raise RuntimeError("Jina API unreachable after all retries")
        print(f"  embedding API unavailable ({batch_label}, {status}), "
              f"retrying in {int(delay)}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(delay)
        delay *= 2


def _embed_api(texts: list[str], task: str) -> list[list[float]]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {CFG.jina_api_key}",
    }
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        payload = {
            "model": CFG.jina_model,
            "input": batch,
            "task": task,
        }
        resp = _retry_until_success(
            lambda: requests.post(CFG.jina_api_url, headers=headers, json=payload, timeout=60),
            CFG.embed_retries,
            CFG.embed_retry_backoff,
            batch_label=f"batch {i // _BATCH_SIZE + 1}",
        )
        data = resp.json()
        try:
            batch_embeddings = [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unrecognized Jina API response shape, keys={list(data.keys())}. "
                "Inspect the raw response and adjust _embed_api() in embeddings.py."
            )
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def _embed(texts: list[str], task: str) -> list[list[float]]:
    if CFG.embed_backend == "local":
        return _embed_local(texts, task)
    return _embed_api(texts, task)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, task="retrieval.passage")


def embed_query(text: str) -> list[float]:
    return _embed([text], task="retrieval.query")[0]
