import html

import streamlit as st

from config import groq_api_key, set_runtime_keys, tavily_api_key
from text_clean import dedupe_sources, sanitize_reading_text, sanitize_report_markdown

st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [class*="css"]{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.stApp{
    background:
        radial-gradient(circle at top left,#22254b 0%,#0E1117 35%),
        radial-gradient(circle at bottom right,#24183d 0%,#0E1117 35%);
}
section[data-testid="stSidebar"]{ background:#10151F; border-right:1px solid #2d3442; }
.main-title{ text-align:center; font-size:42px; font-weight:700; color:white; margin-top:8px; }
.subtitle{ text-align:center; font-size:17px; color:#C4CBD8; margin-bottom:18px; }
.panel{
    background:#161B22; padding:16px 18px; border-radius:14px;
    border:1px solid #2F3847; margin-bottom:12px;
}
.q-title{ font-size:1.05rem; font-weight:600; margin-bottom:8px; color:#E8EDF5; }
.q-body{ color:#D5DCE8; line-height:1.65; margin-bottom:10px; }
.source-chip{
    display:inline-block; margin:4px 8px 4px 0; padding:6px 10px;
    border-radius:999px; background:#1F2937; border:1px solid #374151; font-size:0.85rem;
}
div.stButton > button{
    width:100%; height:54px; font-size:17px; font-weight:600; border:none;
    border-radius:12px; background:linear-gradient(90deg,#5B8DEF,#8B5CF6); color:white;
}
</style>
""",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.markdown("## 🤖 AI Research Assistant")
    st.markdown("---")
    st.markdown(
        """
**Agents:** Planner → Search → Reader → Writer → Critic

**Fast mode (default):** fewer sources, parallel fetch, structured Q&A UI.
"""
    )
    fast_mode = st.toggle("⚡ Fast mode", value=True)
    deep_review = st.toggle("⭐ Deep LLM review", value=False, disabled=fast_mode)
    st.markdown("---")
    st.markdown("### API keys (optional)")
    groq_in = st.text_input("GROQ_API_KEY", value=groq_api_key(), type="password")
    tavily_in = st.text_input("TAVILY_API_KEY", value=tavily_api_key(), type="password")
    set_runtime_keys(groq=groq_in, tavily=tavily_in)
    if groq_in and tavily_in:
        st.success("Groq + Tavily ready")
    elif groq_in:
        st.info("Groq ready")
    else:
        st.caption("Works without keys (extractive mode)")

st.markdown('<div class="main-title">🤖 Multi-Agent AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Structured research output — clean answers, sources, and report tabs.</div>',
    unsafe_allow_html=True,
)

last = st.session_state.result
c1, c2, c3, c4 = st.columns(4)
c1.metric("Agents", "5")
c2.metric("LLM", (last or {}).get("model") or ("Groq" if groq_api_key() else "Extractive"))
c3.metric("Search", "Tavily" if tavily_api_key() else "DuckDuckGo")
c4.metric("Status", (last or {}).get("status", "Ready").title())

topic = st.text_input(
    "🔎 Research topic",
    placeholder="Example: Quantum computing future",
)

run = st.button("🚀 Generate Research Report", use_container_width=True)

if run:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        progress = st.progress(0)
        status = st.empty()

        def on_progress(pct: int, message: str) -> None:
            progress.progress(min(max(pct, 0), 100))
            status.info(message)

        with st.spinner("Running agents…"):
            from pipeline import run_research_pipeline

            result = run_research_pipeline(
                topic.strip(),
                progress=on_progress,
                fast_mode=fast_mode and not deep_review,
            )
        result["topic"] = topic.strip()
        st.session_state.result = result
        last = result

        if result.get("status") == "completed":
            status.success("✅ Research complete — open the tabs below.")
        else:
            status.error("Research did not complete.")
            for err in result.get("errors") or []:
                st.error(err)

last = st.session_state.result
if last and last.get("status") == "completed":
    hits = last.get("search_hits") or []
    pages = [p for p in (last.get("scraped_pages") or []) if p.get("ok")]
    answers = last.get("answers") or []
    report = sanitize_report_markdown((last.get("report") or "").strip())
    feedback = (last.get("feedback") or "").strip()

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sources", len(hits))
    m2.metric("Pages read", len(pages))
    m3.metric("Questions", len(answers))
    m4.metric("Mode", "Fast" if last.get("fast_mode", True) else "Deep")

    tab_overview, tab_qa, tab_report, tab_sources, tab_review = st.tabs(
        ["📌 Overview", "🧩 Q&A", "📄 Report", "🔗 Sources", "⭐ Review"]
    )

    with tab_overview:
        with st.container(border=True):
            st.markdown(f"### Topic: {last.get('topic') or ''}")
            if answers:
                lead = sanitize_reading_text(answers[0].get("answer") or "", max_len=500)
                st.markdown("**Executive summary**")
                st.write(lead or "Summary unavailable.")

        if answers:
            st.markdown("**Research plan answered**")
            for idx, item in enumerate(answers, start=1):
                st.markdown(f"{idx}. {item.get('question') or ''}")

    with tab_qa:
        for idx, item in enumerate(answers, start=1):
            sources = dedupe_sources(item.get("sources") or [])
            body = sanitize_reading_text(item.get("answer") or "", max_len=900)
            with st.container(border=True):
                st.markdown(f"**Q{idx}. {item.get('question') or ''}**")
                st.write(body)
                if sources:
                    cols = st.columns(min(3, len(sources)))
                    for col, src in zip(cols, sources):
                        with col:
                            st.link_button(src["title"][:48], src["url"], use_container_width=True)

    with tab_report:
        if report:
            st.markdown(report)
            st.download_button(
                "📥 Download report",
                data=report,
                file_name=f"{(last.get('topic') or 'research').replace(' ', '_')}_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("No report text available.")

    with tab_sources:
        st.caption("Unique links collected by the Search and Reader agents.")
        for hit in hits[:20]:
            title = sanitize_reading_text(hit.get("title") or "Source", max_len=100)
            url = hit.get("url") or ""
            snippet = sanitize_reading_text(hit.get("snippet") or "", max_len=220)
            st.markdown(f"**{title}** · `{hit.get('source', 'web')}`")
            st.link_button("Open source", url, use_container_width=False)
            if snippet:
                st.write(snippet)
            st.divider()

    with tab_review:
        if feedback:
            st.markdown(feedback)
        else:
            st.info("No critic review for this run.")

    with st.expander("📋 Agent log", expanded=False):
        for line in last.get("logs") or []:
            st.write(f"- {line}")

st.markdown("---")
st.caption("Multi-agent orchestration · Groq · Tavily · DuckDuckGo · Wikipedia · Streamlit")
