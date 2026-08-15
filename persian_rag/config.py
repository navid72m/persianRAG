import os
from dataclasses import dataclass
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_PATH)


def _secret(name: str, default=""):
    """Resolve a secret: real env var first, then Streamlit secrets
    (deployed on Streamlit Community Cloud), then nothing."""
    v = os.getenv(name)
    if v:
        return v
    try:
        import streamlit as st

        v = st.secrets.get(name)
        if v is not None:
            return v
    except Exception:
        pass
    return default


@dataclass(frozen=True)
class Config:
    # --- credentials ---
    # LLM calls (router, rerank, generation) go through the OpenCode Go
    # gateway (OpenAI-compatible) using the OpenCode Go API key.
    llm_api_key: str = _secret("OPENCODE_API_KEY")
    llm_base_url: str = _secret("LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
    qdrant_url: str = _secret("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = _secret("QDRANT_API_KEY") or None

    # --- embeddings: Jina Embeddings v3 ---
    # Multilingual (incl. Persian), instruction-tuned with task prompts
    # ("retrieval.query" / "retrieval.passage"), 1024 dims.
    #
    # Backend "local"  = sentence-transformers running the model locally.
    # Backend "api"    = https://api.jina.ai (needs JINA_API_KEY; fast).
    # NOTE: switching models invalidates existing Qdrant vectors — re-run
    # `python -m persian_rag.resume_embed` after a model change.
    embed_backend: str = _secret("EMBED_BACKEND", "local")
    embed_model: str = _secret("EMBED_MODEL", "heydariAI/persian-embeddings")
    embed_device: str = _secret("EMBED_DEVICE", "cpu")  # cuda | mps | cpu (cuda auto-falls back)

    jina_api_key: str = _secret("JINA_API_KEY")
    jina_api_url: str = _secret("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
    jina_model: str = _secret("JINA_MODEL", "jina-embeddings-v3")
    embed_retries: int = 6  # connection/timeout/5xx retries per batch (API backend)
    embed_retry_backoff: float = 20.0  # seconds, doubles each retry (20, 40, 80, ...)

    embed_dim: int = 1024  # jina-embeddings-v3; collection is created with this size

    # --- LLM models (OpenCode Go, DeepSeek) ---
    router_llm: str = "deepseek-v4-flash"      # rewrite/routing + LLM-based rerank
    generation_llm: str = "deepseek-v4-flash"       # final grounded answer

    # --- chunking (units: whitespace-split words, see chunking.py) ---
    child_chunk_tokens: int = 260   # ~350-400 LLM tokens in Persian
    child_chunk_overlap: int = 45
    parent_chunk_tokens: int = 1300  # ~1800 LLM tokens in Persian

    # --- OCR fallback for scanned/image PDFs (no text layer) ---
    ocr_enabled: bool = True
    ocr_lang: str = "fas+eng"  # tesseract language pack(s); `fas` = Persian
    ocr_dpi: int = 300  # higher = better Persian accuracy, slower

    # --- retrieval ---
    collection_name: str = "persian_doc_children"
    parent_db_path: str = _secret("PARENT_DB_PATH", "parents.sqlite")
    query_log_path: str = _secret("QUERY_LOG_PATH", "queries.log")
    retrieve_top_k: int = 20          # per (sub-)query, before rerank
    rerank_top_k: int = 5             # parents kept for generation
    max_sub_queries: int = 4


CFG = Config()
