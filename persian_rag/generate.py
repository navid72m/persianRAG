from openai import OpenAI

from .config import CFG

_client = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=CFG.llm_api_key, base_url=CFG.llm_base_url)
    return _client

_SYSTEM_PROMPT = """\
تو دستیار پرسش‌وپاسخ روی یک سند فارسی هستی. فقط بر اساس قطعات متن ارائه‌شده پاسخ بده.
اگر پاسخ در متن‌های ارائه‌شده وجود ندارد، صادقانه بگو که در سند یافت نشد؛ حدس نزن.
در پایان پاسخ، شماره صفحات مرتبط را ذکر کن.
"""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        context_block = "(هیچ قطعه مرتبطی یافت نشد)"
    else:
        context_block = "\n\n---\n\n".join(
            f"[صفحات {c['page_start']}-{c['page_end']}]\n{c['text']}"
            for c in context_chunks
        )

    resp = _client_singleton().chat.completions.create(
        model=CFG.generation_llm,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"پرسش: {query}\n\nقطعات مرتبط از سند:\n{context_block}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def generate_direct_answer(query: str, chat_history: list[dict] | None = None) -> str:
    """Used for chitchat/out-of-scope route — no retrieval context."""
    messages = [{"role": "system", "content": "به فارسی و مختصر پاسخ بده. اگر پرسش خارج از حوزه سند مرجع است، این را شفاف بگو."}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": query})
    resp = _client_singleton().chat.completions.create(model=CFG.router_llm, messages=messages, temperature=0.3)
    return resp.choices[0].message.content
