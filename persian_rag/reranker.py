"""
LLM-based rerank, replacing a dedicated rerank API. Scores each candidate
parent chunk's relevance to the query with one structured-output call to
the cheap router model, rather than a purpose-built cross-encoder — slower
and pricier per query than a dedicated reranker, but keeps the stack down
to one provider (OpenAI).
"""
import json

from openai import OpenAI

from .config import CFG

_client = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=CFG.llm_api_key, base_url=CFG.llm_base_url)
    return _client


_SYSTEM_PROMPT = """\
تو یک ماژول رتبه‌بندی مجدد (rerank) برای یک سامانه بازیابی اطلاعات فارسی هستی.
به تو یک پرسش و چند قطعه متن نامزد داده می‌شود. برای هر قطعه، میزان ارتباط آن
با پرسش را از ۰ تا ۱۰۰ امتیاز بده (۱۰۰ یعنی کاملاً مرتبط و پاسخ‌گو، ۰ یعنی کاملاً نامرتبط).

خروجی باید فقط یک JSON معتبر با این ساختار باشد، بدون هیچ متن اضافه:
{"scores": [{"index": 0, "relevance_score": 0}, {"index": 1, "relevance_score": 0}, ...]}

باید دقیقاً به تعداد قطعات ورودی، یک آیتم در آرایه scores برگردانی، به همان ترتیب index.
"""


def rerank(query: str, documents: list[str], top_n: int) -> list[dict]:
    """Returns [{index, relevance_score}, ...] sorted best-first, length top_n.
    Mirrors the shape of the previous Cohere-based rerank() so callers
    (retrieval.py) don't need to change.
    """
    if not documents:
        return []

    numbered = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents))
    user_content = f"پرسش: {query}\n\nقطعات نامزد:\n{numbered}"

    resp = _client_singleton().chat.completions.create(
        model=CFG.router_llm,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
        scores = parsed["scores"]
    except (json.JSONDecodeError, KeyError, TypeError):
        # fail safe: preserve original hybrid-search order rather than crash
        return [{"index": i, "relevance_score": 0.0} for i in range(min(top_n, len(documents)))]

    scores.sort(key=lambda s: s.get("relevance_score", 0), reverse=True)
    return scores[:top_n]