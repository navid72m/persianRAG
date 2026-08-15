"""
Streamlit chat UI for the Persian RAG pipeline. No document upload here on
purpose — ingestion is a separate offline step (`python -m persian_rag.ingest`).

Run with:
    streamlit run app.py
"""
import json
import time
import uuid
from datetime import datetime, timezone

import streamlit as st

from persian_rag.config import CFG
from persian_rag.rag import run_query_stream

st.set_page_config(page_title="پرسش‌وپاسخ سند", page_icon="📄", layout="centered")

# ---------------------------------------------------------------------------
# Styling (minimal)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    direction: rtl;
    font-family: "Vazirmatn", "Tahoma", sans-serif;
    background: #ffffff;
}

.stApp, .stChatMessage, .stMarkdown, textarea, input {
    font-family: "Vazirmatn", "Tahoma", sans-serif;
}

/* header */
.hero {
    direction: rtl;
    text-align: center;
    padding: 1.5rem 0 0.3rem;
}
.hero h1 {
    font-size: 1.7rem;
    font-weight: 800;
    color: #111827;
    margin: 0;
}
.hero .sub {
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.1rem 0;
}
[data-testid="stChatMessageContent"] {
    direction: rtl;
    text-align: right;
    line-height: 1.9;
    color: #1f2937;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #f3f4f6;
    border-radius: 12px;
    padding: 0.6rem 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: transparent;
    padding: 0.3rem 0.1rem;
}

/* route badge */
.route-badge {
    direction: ltr;
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    margin-bottom: 0.3rem;
    background: #f1f5f9;
    color: #475569;
}
.route-badge.simple_retrieval { background: #e0f2fe; color: #0369a1; }
.route-badge.multi_hop       { background: #f3e8ff; color: #7e22ce; }
.route-badge.direct_answer   { background: #dcfce7; color: #15803d; }

/* source cards */
.src-card {
    direction: rtl;
    border-bottom: 1px solid #e5e7eb;
    padding: 0.55rem 0.1rem;
}
.src-card .pages {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    color: #0f766e;
    direction: ltr;
    margin-bottom: 0.2rem;
}
.src-card .txt {
    color: #6b7280;
    font-size: 0.8rem;
    line-height: 1.8;
}

/* chat input */
[data-testid="stChatInput"] {
    direction: rtl;
    border: 1px solid #d1d5db;
    border-radius: 12px;
}
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }

/* typing indicator */
.typing { direction: rtl; color: #9ca3af; font-size: 0.85rem; }
.typing .dot { display: inline-block; animation: blink 1.2s infinite; }
.typing .dot:nth-child(2) { animation-delay: 0.2s; }
.typing .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }

/* expander */
[data-testid="stExpander"] summary { direction: rtl; font-weight: 600; color: #1f2937; }

.footer {
    direction: rtl;
    text-align: center;
    color: #9ca3af;
    font-size: 0.7rem;
    margin-top: 2.5rem;
}

/* stage tracker */
.stage-tracker {
    direction: rtl;
    font-size: 0.85rem;
    padding: 0.4rem 0;
}
.stage-tracker .st-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.22rem 0;
    color: #9ca3af;
}
.stage-tracker .st-row.done { color: #16a34a; }
.stage-tracker .st-row.active {
    color: #1f2937;
    font-weight: 700;
}
.stage-tracker .st-ico { width: 1.2rem; text-align: center; }
.stage-tracker .st-note { font-size: 0.72rem; color: #9ca3af; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>📄 پرسش‌وپاسخ روی سند</h1>
  <div class="sub">بازیابی ترکیبی · بازنویسی پرسش · مسیریابی · چانک والد-فرزند</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta?}]

# ---------------------------------------------------------------------------
# Inline controls (no sidebar)
# ---------------------------------------------------------------------------
col_a, col_b, col_c = st.columns([1, 1, 1], vertical_alignment="center")
with col_a:
    if st.button("🗑️ پاک‌کردن گفتگو", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col_b:
    show_sources = st.checkbox("نمایش منابع", value=True)
with col_c:
    show_route = st.checkbox("نمایش مسیر و نیت", value=True)

_ROUTE_FA = {
    "direct_answer": "پاسخ مستقیم",
    "simple_retrieval": "بازیابی ساده",
    "multi_hop": "چندمرحله‌ای",
}

_STAGES = [
    ("rewrite_and_classify", "🧠", "بازنویسی و تحلیل پرسش"),
    ("retrieve", "🔎", "جست‌وجو در سند"),
    ("generate", "✍️", "تولید پاسخ"),
    ("direct_answer", "💬", "نوشتن پاسخ"),
]
_STAGE_NOTES = {
    "retrieve": "بازیابی روی سرور انجام می‌شود و ممکن است کمی طول بکشد…",
}


def _render_tracker(ph, done: list[str], current: str) -> None:
    rows = []
    for node, icon, label in _STAGES:
        if node in done:
            cls, ico = "done", "✓"
        elif node == current:
            cls, ico = "active", "●"
        else:
            cls, ico = "", "·"
        note = ""
        if node == current and node in _STAGE_NOTES:
            note = f'<div class="st-note">{_STAGE_NOTES[node]}</div>'
        rows.append(
            f'<div class="st-row {cls}"><span class="st-ico">{ico}</span>'
            f'<span>{label}</span></div>{note}'
        )
    ph.markdown(f'<div class="stage-tracker">{"".join(rows)}</div>', unsafe_allow_html=True)


def _fmt_elapsed(secs: float) -> str:
    if secs < 60:
        return f"{int(secs)} ثانیه"
    return f"{int(secs // 60)} دقیقه و {int(secs % 60)} ثانیه"


def _math_clean(text: str) -> str:
    """Convert LaTeX math delimiters to Streamlit-compatible ones:
    \\(...\\) -> $...$, \\[...\\] -> $$...$$. Leaves existing $..$ untouched."""
    import re

    # display: \[ ... \]
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", lambda m: "$$\n" + m.group(1) + "\n$$", text, flags=re.S)
    # inline: \( ... \)
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", lambda m: "$" + m.group(1) + "$", text, flags=re.S)
    return text


def _log_query(entry: dict) -> None:
    """Append one JSON line per query to QUERY_LOG_PATH (visible on the server)."""
    try:
        entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(CFG.query_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the chat


def _route_badge(meta: dict) -> str:
    route = meta.get("route") or "—"
    intent = meta.get("intent") or ""
    label = _ROUTE_FA.get(route, route)
    return (f'<span class="route-badge {route}">{label} · {intent}</span>')


def _render_sources(retrieved: list[dict]) -> None:
    with st.expander(f"📚 {len(retrieved)} قطعه مرتبط از سند", expanded=False):
        for c in retrieved:
            txt = c.get("text", "")
            pages = f"{c.get('page_start', '?')}–{c.get('page_end', '?')}"
            st.markdown(
                f'<div class="src-card">'
                f'<div class="pages">صفحات {pages}</div>'
                f'<div class="txt">{txt[:450]}{"…" if len(txt) > 450 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            meta = msg.get("meta") or {}
            if show_route and meta.get("route"):
                st.markdown(_route_badge(meta), unsafe_allow_html=True)
            st.markdown(_math_clean(msg["content"]))
            if show_sources and meta.get("retrieved"):
                _render_sources(meta["retrieved"])
        else:
            st.markdown(msg["content"])


# replay history
for msg in st.session_state.messages:
    _render_message(msg)


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------
def _history_for_pipeline() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


user_input = st.chat_input("پرسش خود را بنویسید...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    _render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        tracker = st.empty()
        bar = st.empty()
        done_stages: list[str] = []
        t0 = time.time()

        def on_update(node: str, _partial: dict) -> None:
            if node not in done_stages:
                done_stages.append(node)
            _render_tracker(tracker, done_stages, node)
            bar.progress(min(len(done_stages) / len(_STAGES), 1.0))

        try:
            history = _history_for_pipeline()[:-1]  # exclude the message just added
            state = run_query_stream(user_input, chat_history=history, on_update=on_update)
            answer = state.get("answer", "پاسخی تولید نشد.")
            meta = {
                "route": state.get("route"),
                "intent": state.get("intent"),
                "retrieved": state.get("retrieved", []),
            }
            error = None
        except Exception as e:
            answer = f"⚠️ خطا در پردازش پرسش: `{e}`"
            meta = {}
            error = str(e)

        elapsed_s = round(time.time() - t0, 1)
        st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
        _log_query({
            "session": st.session_state.session_id,
            "query": user_input,
            "route": meta.get("route"),
            "intent": meta.get("intent"),
            "elapsed_s": elapsed_s,
            "chunks": len(meta.get("retrieved", [])),
            "error": error,
        })

        bar.empty()
        tracker.empty()
        if meta.get("route") and show_route:
            st.markdown(_route_badge(meta), unsafe_allow_html=True)
        st.markdown(_math_clean(answer))
        if meta.get("retrieved") and show_sources:
            _render_sources(meta["retrieved"])
        if not isinstance(answer, str) or not answer.startswith("⚠️"):
            st.caption(f"⏱ زمان پاسخ: {_fmt_elapsed(elapsed_s)}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})

st.markdown(
    '<div class="footer">این رابط فقط برای پرسیدن سؤال است — سند از پیش ایندکس شده است.</div>',
    unsafe_allow_html=True,
)
