import html

import streamlit as st

from config import groq_api_key, set_runtime_keys, tavily_api_key
from pipeline import run_research_pipeline

st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]{ font-family:'Poppins',sans-serif; }
.stApp{
    background:
        radial-gradient(circle at top left,#22254b 0%,#0E1117 35%),
        radial-gradient(circle at bottom right,#24183d 0%,#0E1117 35%);
    color:white;
}
section[data-testid="stSidebar"]{ background:#10151F; border-right:1px solid #2d3442; }
.main-title{ text-align:center; font-size:46px; font-weight:700; color:white; margin-top:10px; }
.subtitle{ text-align:center; font-size:18px; color:#C4CBD8; margin-bottom:24px; }
.card, .metric-card{
    background:#161B22; padding:18px; border-radius:16px;
    border:1px solid #2F3847; box-shadow:0px 6px 18px rgba(0,0,0,.35);
}
.report-card{
    background:#171F2C; padding:25px; border-radius:16px;
    border-left:6px solid #5B8DEF; color:white; line-height:1.8;
}
.feedback-card{
    background:#221C35; padding:25px; border-radius:16px;
    border-left:6px solid #A855F7; color:white; line-height:1.8;
}
.answer-card{
    background:#162033; padding:18px; border-radius:14px;
    border:1px solid #2F3847; margin-bottom:12px; line-height:1.7;
}
div.stButton > button{
    width:100%; height:58px; font-size:18px; font-weight:600; border:none;
    border-radius:12px; background:linear-gradient(90deg,#5B8DEF,#8B5CF6); color:white;
}
hr{ border:0; border-top:1px solid #2F3847; }
</style>
""",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=120)
    st.title("AI Research Assistant")
    st.markdown("---")
    st.markdown("## ⚙️ Workflow")
    st.markdown(
        """
🧭 **Planner Agent** — research questions

🔍 **Search Agent** — Tavily / DuckDuckGo / Wikipedia

🌐 **Reader Agent** — scrape source pages

📝 **Writer Agent** — answer every question

⭐ **Critic Agent** — score and review
"""
    )
    st.markdown("---")
    st.markdown("### API keys (optional)")
    groq_in = st.text_input("GROQ_API_KEY", value=groq_api_key(), type="password")
    tavily_in = st.text_input("TAVILY_API_KEY", value=tavily_api_key(), type="password")
    set_runtime_keys(groq=groq_in, tavily=tavily_in)
    if groq_in and tavily_in:
        st.success("Groq + Tavily configured")
    elif groq_in:
        st.info("Groq on · search will use DuckDuckGo/Wikipedia")
    else:
        st.warning("No Groq key · extractive report from live sources still runs")
    st.caption("Powered by LangGraph multi-agent orchestration")

st.markdown('<div class="main-title">🤖 Multi-Agent AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Each query is planned, searched, read, answered, and reviewed — not marked done while empty.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Agents", "5")
c2.metric("LLM", "Groq" if groq_api_key() else "Extractive")
c3.metric("Search", "Tavily" if tavily_api_key() else "DuckDuckGo")
last = st.session_state.result
c4.metric("Status", (last or {}).get("status", "Ready").title())

topic = st.text_input(
    "🔎 Enter your research topic or question",
    placeholder="Example: Quantum Computing Future",
)

run = st.button("🚀 Generate Research Report", use_container_width=True)

if run:
    if not topic.strip():
        st.warning("⚠️ Please enter a research topic.")
    else:
        progress = st.progress(0)
        status = st.empty()

        def on_progress(pct: int, message: str) -> None:
            progress.progress(min(max(pct, 0), 100))
            status.info(message)

        with st.spinner("Running Planner → Search → Reader → Writer → Critic..."):
            result = run_research_pipeline(topic.strip(), progress=on_progress)
        st.session_state.result = result
        last = result

        if result.get("status") == "completed" and result.get("report") and result.get("search_hits"):
            status.success("✅ Research completed with sources and answers.")
        else:
            status.error("❌ Research did not produce a complete result.")
            for err in result.get("errors") or []:
                st.error(err)

last = st.session_state.result
if last:
    st.write("")
    m1, m2, m3 = st.columns(3)
    hits = last.get("search_hits") or []
    pages = last.get("scraped_pages") or []
    readable = [page for page in pages if page.get("ok")]
    m1.metric("Sources found", str(len(hits)))
    m2.metric("Pages read", str(len(readable)))
    m3.metric("Model", last.get("model") or "—")

    answers = last.get("answers") or []
    if answers:
        st.markdown("## 🧩 Answers to each research question")
        for item in answers:
            sources = item.get("sources") or []
            source_html = "".join(
                f'<div>• <a href="{html.escape(src.get("url", ""), quote=True)}" target="_blank">'
                f'{html.escape(src.get("title") or src.get("url") or "")}</a></div>'
                for src in sources
            )
            st.markdown(f"#### {item.get('question') or ''}")
            st.markdown(item.get("answer") or "")
            if source_html:
                st.markdown(source_html, unsafe_allow_html=True)
            st.markdown("---")

    report = (last.get("report") or "").strip()
    if report:
        st.markdown("## 📄 Research Report")
        st.markdown(report)
        st.download_button(
            "📥 Download Report",
            data=report,
            file_name=f"{(last.get('topic') or 'research').replace(' ', '_')}_Research_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    elif last.get("status") == "completed":
        st.error("The pipeline claimed success but the report was empty.")

    feedback = (last.get("feedback") or "").strip()
    if feedback:
        st.markdown("## ⭐ Critic Review")
        st.markdown(feedback)

    with st.expander("🔍 Search Results", expanded=not hits):
        if hits:
            for hit in hits:
                st.markdown(f"**{hit.get('title') or 'Untitled'}** · {hit.get('source')}  \n{hit.get('url')}  \n{hit.get('snippet')}")
                st.markdown("---")
        else:
            st.warning("No search results available.")

    with st.expander("🌐 Scraped Content", expanded=not readable):
        if readable:
            for page in readable:
                st.markdown(f"**{page.get('title') or page.get('url')}**  \n{page.get('url')}  \n{page.get('text', '')[:1500]}")
                st.markdown("---")
        else:
            st.warning("No scraped content available.")

    with st.expander("📋 Agent log"):
        for line in last.get("logs") or []:
            st.write(f"- {line}")

st.markdown("---")
st.markdown(
    """
<div style="text-align:center;color:#AAB3C5;">
Built with <b>LangGraph</b> multi-agent orchestration · Groq · Tavily · DuckDuckGo · Wikipedia · Streamlit
</div>
""",
    unsafe_allow_html=True,
)
