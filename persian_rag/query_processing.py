"""Query rewrite + intent detection, driven by one structured-output call
to a small/cheap model. This is Stage 1 of the pipeline: it decides *what
the user actually wants* and *how the rest of the graph should route*.
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
تو یک ماژول پردازش پرسش برای یک سامانه بازیابی اطلاعات (RAG) روی یک سند فارسی هستی.
برای هر پرسش کاربر باید خروجی JSON زیر را تولید کنی، بدون هیچ متن اضافه:

{
  "intent": one of ["factual_lookup", "summarization", "comparison", "definition", "chitchat", "out_of_scope"],
  "needs_retrieval": boolean,
  "rewritten_query": string,   // پرسش بازنویسی‌شده: حل ارجاعات ضمیری با تاریخچه گفتگو، رفع ابهام، عادی‌سازی
  "sub_queries": [string],     // اگر پرسش چندبخشی یا مقایسه‌ای است، به ۲ تا ۴ زیرپرسش مستقل تجزیه کن؛ در غیر این صورت آرایه خالی
  "route": one of ["direct_answer", "simple_retrieval", "multi_hop"]
}

قواعد مسیریابی:
- اگر پرسش گفتگوی معمولی یا کاملاً خارج از موضوع سند است -> route="direct_answer", needs_retrieval=false
- اگر پرسش ساده و تک‌بخشی است -> route="simple_retrieval", sub_queries خالی
- اگر پرسش شامل مقایسه، چند بخش مجزا، یا نیاز به ترکیب چند بخش از سند دارد -> route="multi_hop", sub_queries را پر کن
- اگر پرسش خلاصه‌سازی کل سند یا بخش بزرگی از آن را می‌خواهد -> intent="summarization", route="multi_hop"

فقط JSON معتبر برگردان.
"""


def process_query(user_query: str, chat_history: list[dict] | None = None) -> dict:
    history_text = ""
    if chat_history:
        history_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in chat_history[-6:])

    user_content = f"تاریخچه گفتگو:\n{history_text}\n\nپرسش فعلی:\n{user_query}"

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
    except json.JSONDecodeError:
        # fail safe: treat as a simple retrieval of the raw query
        parsed = {
            "intent": "factual_lookup",
            "needs_retrieval": True,
            "rewritten_query": user_query,
            "sub_queries": [],
            "route": "simple_retrieval",
        }

    parsed.setdefault("sub_queries", [])
    parsed["sub_queries"] = parsed["sub_queries"][:CFG.max_sub_queries]
    parsed.setdefault("rewritten_query", user_query)
    return parsed
