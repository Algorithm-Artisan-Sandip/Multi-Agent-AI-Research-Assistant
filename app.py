import html
import json
import re

import streamlit as st

from config import integration_status
from text_clean import dedupe_sources, sanitize_reading_text, sanitize_report_markdown

st.set_page_config(
    page_title="AI Research Assistant Pro",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #1e2a5a 0%, transparent 55%),
              radial-gradient(900px 500px at 100% 0%, #3b1d60 0%, transparent 50%),
              #0b0f17;
  color: #e8edf7;
}
section[data-testid="stSidebar"] {
  background: rgba(12, 16, 26, 0.92);
  border-right: 1px solid rgba(255,255,255,0.08);
}
.hero {
  padding: 28px 28px 22px;
  border-radius: 20px;
  border: 1px solid rgba(255,255,255,0.1);
  background: linear-gradient(135deg, rgba(91,141,239,0.18), rgba(139,92,246,0.14));
  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
  margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero p { margin: 8px 0 0; color: #c6d0e4; }
.badge {
  display:inline-block; padding:6px 12px; margin:4px 6px 0 0;
  border-radius:999px; font-size:0.78rem; font-weight:600;
  border:1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06);
}
.badge.ok { color:#86efac; border-color: rgba(134,239,172,0.35); }
.badge.dim { color:#cbd5e1; }
.flow-wrap { margin-top: 8px; }
.flow-node {
  margin: 0 auto 6px; padding: 10px 12px; max-width: 220px;
  border-radius: 12px; text-align: center; font-size: 0.82rem; font-weight: 600;
  border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.05);
}
.flow-node.planner { border-color: rgba(91,141,239,0.45); }
.flow-node.search { border-color: rgba(56,189,248,0.45); }
.flow-node.reader { border-color: rgba(52,211,153,0.45); }
.flow-node.writer { border-color: rgba(167,139,250,0.45); }
.flow-node.critic { border-color: rgba(251,191,36,0.45); }
.flow-arrow { text-align:center; color:#94a3b8; font-size: 0.9rem; line-height: 1; margin: 2px 0; }
.flow-note { font-size: 0.72rem; color: #94a3b8; text-align: center; margin-top: 8px; }
div.stButton > button {
  height: 52px; border-radius: 14px; border: none; font-weight: 600;
  background: linear-gradient(90deg, #5B8DEF, #8B5CF6);
}
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 8px 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "research_topic" not in st.session_state:
    st.session_state.research_topic = ""


def run_research_job(topic: str, *, fast: bool) -> None:
    progress = st.progress(0)
    status_box = st.empty()

    def on_progress(pct: int, message: str) -> None:
        progress.progress(min(max(pct, 0), 100))
        status_box.info(message)

    with st.spinner("Agents running…"):
        from pipeline import run_research_pipeline

        result = run_research_pipeline(topic, progress=on_progress, fast_mode=fast)
    result["topic"] = topic
    st.session_state.result = result
    if result.get("status") == "completed":
        status_box.success("Research complete. Explore the structured tabs below.")
    else:
        status_box.error("Research could not be completed.")
        for err in result.get("errors") or []:
            st.error(err)


status = integration_status()

with st.sidebar:
    st.markdown("### 🛰️ Research Control")
    st.markdown("LangGraph multi-agent pipeline with parallel search, reader, writer, and critic.")
    fast_mode = st.toggle("⚡ Fast mode", value=True, help="Recommended for speed.")
    deep_review = st.toggle("🧠 Deep LLM synthesis", value=False, disabled=fast_mode)
    st.markdown("---")
    st.markdown("**Agent orchestration**")
    st.markdown(
        """
<div class="flow-wrap">
  <div class="flow-node planner">🧭 Planner Agent<br><span style="font-weight:400">Research questions</span></div>
  <div class="flow-arrow">↓</div>
  <div class="flow-node search">🔍 Search Agent<br><span style="font-weight:400">Tavily · DDG · Wiki</span></div>
  <div class="flow-arrow">↓</div>
  <div class="flow-node reader">🌐 Reader Agent<br><span style="font-weight:400">Scrape + media</span></div>
  <div class="flow-arrow">↓</div>
  <div class="flow-node writer">📝 Writer Agent<br><span style="font-weight:400">Q&A + report</span></div>
  <div class="flow-arrow">↓</div>
  <div class="flow-node critic">⭐ Critic Agent<br><span style="font-weight:400">Quality review</span></div>
  <div class="flow-note">LangGraph state machine · fail-fast on empty sources</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Integrations**")
    groq_badge = "ok" if status["groq"] else "dim"
    tavily_badge = "ok" if status["tavily"] else "dim"
    st.markdown(
        f'<span class="badge {groq_badge}">Groq: {"connected" if status["groq"] else "fallback"}</span>'
        f'<span class="badge {tavily_badge}">Tavily: {"connected" if status["tavily"] else "open web"}</span>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero">
  <h1>🛰️ Advanced Multi-Agent Research Assistant</h1>
  <p>Open-topic research with structured reports, media discovery, and source transparency.</p>
</div>
""",
    unsafe_allow_html=True,
)

last = st.session_state.result
c1, c2, c3, c4 = st.columns(4)
c1.metric("Agents", "5")
c2.metric("LLM", (last or {}).get("model") or ("Groq" if status["groq"] else "Extractive"))
c3.metric("Search", "Tavily" if status["tavily"] else "Multi-provider")
c4.metric("Status", (last or {}).get("status", "Ready").title())

examples = [
    "Quantum computing future",
    "Who will win mid term election in USA in 2026",
    "Global semiconductor supply chain 2026",
    "CRISPR therapy clinical trials",
]
st.caption("Try an example topic (fills the field and starts research)")
ex_cols = st.columns(len(examples))
for idx, (col, ex) in enumerate(zip(ex_cols, examples)):
    if col.button(ex, use_container_width=True, key=f"example_topic_{idx}"):
        st.session_state.research_topic = ex
        st.session_state.auto_run_research = True

st.text_input(
    "Research topic",
    key="research_topic",
    placeholder="Any research question or topic — politics, science, markets, history…",
)

run = st.button("Generate research report", type="primary", use_container_width=True)

should_run = run or st.session_state.pop("auto_run_research", False)
if should_run:
    topic_value = (st.session_state.research_topic or "").strip()
    if not topic_value:
        st.warning("Enter a research topic.")
    else:
        run_research_job(topic_value, fast=fast_mode and not deep_review)
        last = st.session_state.result

last = st.session_state.result
if last and last.get("status") == "completed":
    hits = last.get("search_hits") or []
    pages = [p for p in (last.get("scraped_pages") or []) if p.get("ok")]
    answers = last.get("answers") or []
    media = last.get("media") or []
    report = sanitize_report_markdown((last.get("report") or "").strip())
    feedback = (last.get("feedback") or "").strip()
    summary = sanitize_reading_text(
        last.get("executive_summary") or (answers[0].get("answer") if answers else ""),
        max_len=700,
    )

    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sources", len(hits))
    m2.metric("Pages", len(pages))
    m3.metric("Questions", len(answers))
    m4.metric("Media", len(media))
    m5.metric("Synthesis", "Groq" if last.get("used_llm_synthesis") else "Extractive")

    tabs = st.tabs(["Overview", "Q&A", "Report", "Media", "Sources", "Review"])

    with tabs[0]:
        st.subheader(last.get("topic") or "Research topic")
        st.markdown("**Executive summary**")
        st.write(summary or "Summary unavailable.")
        st.markdown("**Research plan**")
        for i, item in enumerate(answers, start=1):
            st.markdown(f"{i}. {item.get('question') or ''}")

    with tabs[1]:
        for i, item in enumerate(answers, start=1):
            sources = dedupe_sources(item.get("sources") or [])
            body = sanitize_reading_text(item.get("answer") or "", max_len=1000)
            with st.container(border=True):
                st.markdown(f"**Q{i}. {item.get('question') or ''}**")
                st.write(body)
                if sources:
                    sc = st.columns(min(3, len(sources)))
                    for col, src in zip(sc, sources):
                        with col:
                            st.link_button(src["title"][:40], src["url"], use_container_width=True)

    with tabs[2]:
        if report:
            st.markdown(report)
            bundle = {
                "topic": last.get("topic"),
                "report": report,
                "answers": answers,
                "media": media,
                "sources": hits,
            }
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download report (Markdown)",
                data=report,
                file_name=f"{(last.get('topic') or 'research').replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            c2.download_button(
                "Download full bundle (JSON)",
                data=json.dumps(bundle, indent=2),
                file_name=f"{(last.get('topic') or 'research').replace(' ', '_')}_bundle.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("No report generated.")

    with tabs[3]:
        if not media:
            st.info("No media assets discovered for this run.")
        else:
            images = [m for m in media if m.get("type") == "image"]
            videos = [m for m in media if m.get("type") == "video"]
            st.caption(f"{len(images)} images · {len(videos)} videos/embeds")
            if images:
                st.markdown("**Images**")
                cols = st.columns(3)
                for idx, item in enumerate(images[:18]):
                    with cols[idx % 3]:
                        st.caption(item.get("title") or "Image")
                        st.image(item["url"], use_container_width=True)
                        st.link_button("Open", item["url"], use_container_width=True, key=f"img_{idx}")
            if videos:
                st.markdown("**Videos**")
                for idx, item in enumerate(videos[:12]):
                    title = item.get("title") or "Video"
                    url = item["url"]
                    st.markdown(f"**{title}**")
                    if "youtube.com" in url or "youtu.be" in url:
                        st.video(url)
                    else:
                        st.link_button("Open video", url, use_container_width=True, key=f"vid_{idx}")
                    st.divider()

    with tabs[4]:
        for hit in hits[:25]:
            title = sanitize_reading_text(hit.get("title") or "Source", max_len=120)
            url = hit.get("url") or ""
            snippet = sanitize_reading_text(hit.get("snippet") or "", max_len=260)
            st.markdown(f"**{title}** · `{hit.get('source', 'web')}`")
            st.link_button("Open", url, use_container_width=False)
            if snippet:
                st.caption(snippet)
            st.divider()

    with tabs[5]:
        if feedback:
            score_match = re.search(r"(\d+(?:\.\d+)?)/10", feedback)
            if score_match:
                st.metric("Quality score", f"{score_match.group(1)}/10")
            st.markdown(feedback)
        else:
            st.info("No review available.")

    with st.expander("Agent log", expanded=False):
        for line in last.get("logs") or []:
            st.write(f"- {line}")

st.caption("LangGraph · LangChain · Groq · Tavily · DuckDuckGo · Wikipedia")
