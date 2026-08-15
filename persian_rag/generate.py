import json
import math

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


# ---------------------------------------------------------------------------
# Formula calculation support
# ---------------------------------------------------------------------------
_CALC_EXTRACT_PROMPT = """\
تو یک ماژول محاسبات فنی برای آیین‌نامه ساختمان هستی. کاربر یک پرسش با مقادیر عددی داده و
می‌خواهد نتیجه محاسبه شود. قطعات مرتبط از سند (که ممکن است شامل متن OCR-شدهٔ فرمول باشد) داده می‌شود.

فرمول مرتبط را به صورت یک عبارت پایتونی معتبر تولید کن که فقط از توابع ریاضی استاندارد
(math.sqrt, math.sin, math.cos, math.tan, math.pow, math.pi, ...) و متغیرها استفاده کند.
اگر متن فرمول در قطعات OCR خراب است، آن را از دانش خودت بر اساس بافت سند بازسازی کن.

خروجی فقط JSON معتبر با این ساختار، بدون هیچ متن اضافه:
{
  "formula": "M / (0.9 * fy * d)",
  "variables": {"M": 150, "fy": 400, "d": 0.5},
  "explanation": "توضیح کوتاه فارسی: فرمول چه چیزی را محاسبه می‌کند و هر متغیر چیست"
}

قواعد:
- متغیرهایی که در پرسش کاربر آمده‌اند را دقیقاً با همان عدد (پس از تبدیل یکا به یک دستگاه سازگار) در "variables" بگذار.
- مقادیری که کاربر نداده را حدس نزن — فقط از مقادیر داده‌شده استفاده کن.
- اگر به متغیری نیاز است که کاربر نداده، آن را با عدد 0 نگذار؛ نامش را در "variables" نیاور و در "explanation" بگو که چه پارامتری کم است.
- اگر اصلاً فرمول مرتبطی قابل استخراج نیست، formula را null بده.
"""

_SAFE_MATH = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}


def _safe_eval(expr: str, variables: dict) -> float:
    """Evaluate a Python expression with only math functions + given variables."""
    ns = {**_SAFE_MATH, **{str(k): float(v) for k, v in variables.items()}}
    return float(eval(expr, {"__builtins__": {}}, ns))


def extract_calculation(query: str, context_chunks: list[dict]) -> dict:
    """Ask the LLM for formula + variable values, then evaluate it safely.
    Returns {formula, variables, result, explanation} or
    {formula: None, ...} if nothing extractable."""
    context_block = "\n\n---\n\n".join(
        f"[صفحات {c['page_start']}-{c['page_end']}]\n{c['text']}"
        for c in context_chunks
    )
    resp = _client_singleton().chat.completions.create(
        model=CFG.router_llm,
        messages=[
            {"role": "system", "content": _CALC_EXTRACT_PROMPT},
            {"role": "user", "content": f"پرسش کاربر:\n{query}\n\nقطعات مرتبط از سند:\n{context_block}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"formula": None, "variables": {}, "result": None, "explanation": ""}

    formula = parsed.get("formula")
    variables = parsed.get("variables") or {}
    explanation = parsed.get("explanation", "")
    if not formula:
        return {"formula": None, "variables": variables, "result": None, "explanation": explanation}

    try:
        result = _safe_eval(formula, variables)
    except Exception:
        return {"formula": None, "variables": variables, "result": None, "explanation": explanation}
    return {
        "formula": formula,
        "variables": variables,
        "result": result,
        "explanation": explanation,
    }


_CALC_ANSWER_PROMPT = """\
تو دستیار محاسبات فنی یک سند فارسی هستی (مبحث نهم مقررات ملی ساختمان).
فرمول، مقادیر متغیرها، و نتیجهٔ محاسبه به تو داده شده است.
پاسخ را مرحله‌به‌مرحله و به فارسی بنویس:
۱) بیان فرمول
۲) جای‌گذاری مقادیر
۳) نتیجه نهایی همراه با یکای مناسب (در صورت مشخص بودن در سند)
۴) اگر پارامتری کم بود، صادقانه بگو که برای محاسبه کامل به آن پارامتر نیاز است.
در پایان، شماره صفحات مرتبط را ذکر کن.
"""


def generate_calculated_answer(query: str, calc: dict, context_chunks: list[dict]) -> str:
    """Final Persian answer for a formula calculation, grounded in the context."""
    if calc.get("result") is None:
        # nothing extractable/computable — fall back to plain grounded answer
        return generate_answer(query, context_chunks)

    pages = ""
    if context_chunks:
        starts = [c.get("page_start") for c in context_chunks if c.get("page_start")]
        ends = [c.get("page_end") for c in context_chunks if c.get("page_end")]
        if starts and ends:
            pages = f"[صفحات {min(starts)}-{max(ends)}]"

    resp = _client_singleton().chat.completions.create(
        model=CFG.generation_llm,
        messages=[
            {"role": "system", "content": _CALC_ANSWER_PROMPT},
            {"role": "user", "content": (
                f"پرسش کاربر: {query}\n\n"
                f"فرمول: {calc.get('formula')}\n"
                f"متغیرها: {json.dumps(calc.get('variables'), ensure_ascii=False)}\n"
                f"نتیجه محاسبه: {calc.get('result')}\n"
                f"توضیح: {calc.get('explanation')}\n"
                f"قطعات سند:\n"
                f"{chr(10).join(c['text'][:800] for c in context_chunks[:2])}"
            )},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return f"{answer}\n\n{pages}".strip()
