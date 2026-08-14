"""
Jina Embeddings v3 client — multilingual (incl. Persian), instruction-tuned.

The API takes a `task` field so the model knows which side of a query/document
pair it's encoding. We use:
  - "retrieval.query"   for the user's question
  - "retrieval.passage" for chunks being indexed

Response envelope is OpenAI-style: {"data": [{"embedding": [...]}], ...}.
Reference: https://jina.ai/embeddings/
"""
import time

import requests

from .config import CFG

_BATCH_SIZE = 32  # conservative; the API doesn't document a hard limit

_RETRYABLE = (
    requests.ConnectionError,
    requests.Timeout,
)


def _retry_until_success(attempt_fn, max_retries: int, backoff: float, batch_label: str):
    """Runs attempt_fn, retrying connection failures / 5xx with exponential
    backoff. The embedding endpoint has proven flaky from some networks;
    retrying a single batch in-place avoids losing the whole ingest."""
    delay = backoff
    for attempt in range(max_retries + 1):
        try:
            resp = attempt_fn()
            if resp.status_code < 500:
                return resp
        except _RETRYABLE:
            pass
        if attempt == max_retries:
            raise
        print(f"  embedding API unavailable ({batch_label}), retrying in {int(delay)}s "
              f"(attempt {attempt + 1}/{max_retries})")
        time.sleep(delay)
        delay *= 2


def _call_jina(texts: list[str], task: str) -> list[list[float]]:
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
        if resp.status_code >= 400:
            raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            batch_embeddings = [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unrecognized Jina API response shape, keys={list(data.keys())}. "
                "Inspect the raw response and adjust _call_jina() in embeddings.py."
            )
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _call_jina(texts, task="retrieval.passage")


def embed_query(text: str) -> list[float]:
    return _call_jina([text], task="retrieval.query")[0]
