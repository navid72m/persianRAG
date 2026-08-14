"""
Streamlit chat UI for the Persian RAG pipeline. No document upload here on
purpose — ingestion is a separate offline step (`python -m persian_rag.ingest`)
since it's meant to run once against your 625-page document, not per-session.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st

from persian_rag.rag import run_query

st.set_page_config(page_title="پرسش‌وپاسخ سند", page_icon="📄", layout="centered")

# RTL + Persian-friendly styling
st.markdown("""
<style>
    .stApp, .stChatMessage, .stMarkdown, textarea, input {
        direction: rtl;
        text-align: right;
        font-family: "Vazirmatn", "Tahoma", sans-serif;
    }
    [data-testid="stChatMessageContent"] { direction: rtl; text-align: right; }
    .stChatInput textarea { direction: rtl; text-align: right; }
    .route-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        background: #f0f0f5;
        color: #555;
        margin-bottom: 6px;
        direction: ltr;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 پرسش‌وپاسخ روی سند")
st.caption("بازیابی ترکیبی (Hybrid) + بازنویسی پرسش + مسیریابی + چانک والد-فرزند")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta?}]

with st.sidebar:
    st.subheader("تنظیمات نشست")
    if st.button("🗑️ پاک‌کردن گفتگو", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    show_sources = st.checkbox("نمایش منابع بازیابی‌شده", value=True)
    show_route = st.checkbox("نمایش مسیر و نیت تشخیص‌داده‌شده", value=True)
    st.divider()
    st.caption(
        "این رابط فقط برای پرسیدن سوال است. سند از پیش با دستور "
        "`python -m persian_rag.ingest` ایندکس شده است."
    )

# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        meta = msg.get("meta")
        if meta and msg["role"] == "assistant" and show_route:
            st.markdown(
                f'<span class="route-badge">route: {meta.get("route")} · intent: {meta.get("intent")}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if meta and msg["role"] == "assistant" and show_sources and meta.get("retrieved"):
            with st.expander(f"📚 {len(meta['retrieved'])} قطعه منبع"):
                for c in meta["retrieved"]:
                    st.markdown(f"**صفحات {c['page_start']}-{c['page_end']}**")
                    st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                    st.divider()

# chat history passed to the pipeline for coreference resolution in rewrite step
def _history_for_pipeline() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


user_input = st.chat_input("پرسش خود را بنویسید...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("در حال جست‌وجو و تولید پاسخ...")
        try:
            history = _history_for_pipeline()[:-1]  # exclude the message just added
            state = run_query(user_input, chat_history=history)
            answer = state.get("answer", "پاسخی تولید نشد.")
            meta = {
                "route": state.get("route"),
                "intent": state.get("intent"),
                "retrieved": state.get("retrieved", []),
            }
        except Exception as e:
            answer = f"خطا در پردازش پرسش: `{e}`"
            meta = {}

        placeholder.empty()
        if meta.get("route") and show_route:
            st.markdown(
                f'<span class="route-badge">route: {meta.get("route")} · intent: {meta.get("intent")}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(answer)
        if meta.get("retrieved") and show_sources:
            with st.expander(f"📚 {len(meta['retrieved'])} قطعه منبع"):
                for c in meta["retrieved"]:
                    st.markdown(f"**صفحات {c['page_start']}-{c['page_end']}**")
                    st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})