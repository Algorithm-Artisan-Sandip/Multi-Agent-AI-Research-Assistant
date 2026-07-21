import streamlit as st
from pipeline import run_research_pipeline
import time
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Background */

.stApp{
    background:
        radial-gradient(circle at top left,#22254b 0%,#0E1117 35%),
        radial-gradient(circle at bottom right,#24183d 0%,#0E1117 35%);
    color:white;
}


/* Sidebar */

section[data-testid="stSidebar"]{
    background:#10151F;
    border-right:1px solid #2d3442;
}


/* Header */

.main-title{
    text-align:center;
    font-size:50px;
    font-weight:700;
    color:white;
    margin-top:10px;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#C4CBD8;
    margin-bottom:30px;
}


/* Cards */

.card{
    background:#161B22;
    padding:22px;
    border-radius:16px;
    border:1px solid #2F3847;
    box-shadow:0px 6px 18px rgba(0,0,0,.35);
}

.report-card{
    background:#171F2C;
    padding:25px;
    border-radius:16px;
    border-left:6px solid #5B8DEF;
    color:white;
    line-height:1.8;
}

.feedback-card{
    background:#221C35;
    padding:25px;
    border-radius:16px;
    border-left:6px solid #A855F7;
    color:white;
    line-height:1.8;
}


/* Buttons */

div.stButton > button{
    width:100%;
    height:58px;
    font-size:18px;
    font-weight:600;
    border:none;
    border-radius:12px;
    background:linear-gradient(90deg,#5B8DEF,#8B5CF6);
    color:white;
    transition:0.3s;
}

div.stButton > button:hover{
    transform:scale(1.02);
    background:linear-gradient(90deg,#4F7FEA,#7C3AED);
}


/* Text Input */

div[data-baseweb="input"] input{
    font-size:18px;
}


/* Metrics */

.metric-card{
    background:#161B22;
    border-radius:15px;
    padding:15px;
    text-align:center;
    border:1px solid #303A4B;
}


/* Sidebar Footer */

.sidebar-footer{
    position:fixed;
    bottom:20px;
    left:18px;
    width:250px;
    text-align:center;
    color:#AAB3C5;
    font-size:14px;
    border-top:1px solid #2D3442;
    padding-top:12px;
}



/* Divider */

hr{
    border:0;
    border-top:1px solid #2F3847;
}


</style>
""",
    unsafe_allow_html=True
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=120,
    )

    st.title("AI Research Assistant")

    st.markdown("---")

    st.markdown("## ⚙️ Workflow")

    st.markdown(
        """
🔍 **Search Agent**

🌐 **Reader Agent**

📝 **Writer Agent**

⭐ **Critic Agent**
"""
    )

    st.markdown("---")

    st.success("Powered by Groq + LangGraph + Tavily")

    st.markdown(
        """

""",
        unsafe_allow_html=True,
    )

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
<div class="main-title">

🤖 Multi-Agent AI Research Assistant

</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="subtitle">

Research • Summarize • Critique using Autonomous AI Agents

</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# TOP METRICS
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Agents", "4")

with c2:
    st.metric("LLM", "Groq")

with c3:
    st.metric("Search", "Tavily")

with c4:
    st.metric("Status", "Ready")

st.write("")

# ==========================================================
# INPUT
# ==========================================================

topic = st.text_input(
    "🔎 Enter your research topic",
    placeholder="Example: Artificial Intelligence in Healthcare"
)

st.write("")

# ==========================================================
# RUN PIPELINE
# ==========================================================

if st.button("🚀 Generate Research Report", use_container_width=True):

    if topic.strip() == "":
        st.warning("⚠️ Please enter a research topic.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    try:

        # ------------------------------------------
        # Step 1
        # ------------------------------------------

        status.info("🔍 Initializing Search Agent...")
        progress.progress(10)
        time.sleep(0.5)

        # ------------------------------------------
        # Step 2
        # ------------------------------------------

        status.info("🌐 Searching reliable sources...")
        progress.progress(25)
        time.sleep(0.6)

        # ------------------------------------------
        # Step 3
        # ------------------------------------------

        status.info("📖 Reading web pages...")
        progress.progress(45)
        time.sleep(0.6)

        # ------------------------------------------
        # Step 4
        # ------------------------------------------

        status.info("📝 Writing research report...")
        progress.progress(70)

        with st.spinner("Running Multi-Agent Pipeline..."):

            result = run_research_pipeline(topic)

        # ------------------------------------------
        # Step 5
        # ------------------------------------------

        status.info("⭐ Critic Agent Reviewing...")
        progress.progress(90)
        time.sleep(0.5)

        progress.progress(100)

        status.success("✅ Research Completed Successfully!")

        st.balloons()

        st.write("")

        # ==========================================================
        # DASHBOARD
        # ==========================================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                """
<div class="metric-card">

<h3>📄 Report</h3>

Generated Successfully

</div>
""",
                unsafe_allow_html=True,
            )

        with c2:

            st.markdown(
                f"""
<div class="metric-card">

<h3>📅 Date</h3>

{datetime.now().strftime("%d %b %Y")}

</div>
""",
                unsafe_allow_html=True,
            )

        with c3:

            st.markdown(
                """
<div class="metric-card">

<h3>🤖 Model</h3>

Groq

</div>
""",
                unsafe_allow_html=True,
            )

        st.write("")

        # ==========================================================
        # REPORT
        # ==========================================================

        if "report" in result:

            st.markdown("## 📄 Research Report")

            st.markdown(
                f"""
<div class="report-card">

{result["report"].replace(chr(10),"<br>")}

</div>
""",
                unsafe_allow_html=True,
            )

        st.write("")

        # ==========================================================
        # CRITIC
        # ==========================================================

        if "feedback" in result:

            st.markdown("## ⭐ Critic Review")

            st.markdown(
                f"""
<div class="feedback-card">

{result["feedback"].replace(chr(10),"<br>")}

</div>
""",
                unsafe_allow_html=True,
            )

        st.write("")

        # ==========================================================
        # DOWNLOAD
        # ==========================================================

        if "report" in result:

            st.download_button(
                "📥 Download Report",
                data=result["report"],
                file_name=f"{topic.replace(' ','_')}_Research_Report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.write("")

        # ==========================================================
        # EXPANDERS
        # ==========================================================

        with st.expander("🔍 Search Results"):

            st.write(result.get("search_results", "No search results available."))

        with st.expander("🌐 Scraped Content"):

            st.write(result.get("scraped_content", "No scraped content available."))

    except Exception as e:

        st.error("❌ Pipeline Execution Failed")

        st.exception(e)

# ==========================================================
# FOOTER
# ==========================================================

st.write("")
st.write("")
st.write("")

st.markdown("---")

col1, col2, col3 = st.columns([1,2,1])

with col2:

    st.markdown(
        """
<div style="text-align:center;color:#AAB3C5;">

### 🚀 Multi-Agent AI Research Assistant

Built using

**Groq • LangChain • LangGraph • Tavily • Streamlit**

Made with ❤️ by <b>Sandip</b>

</div>
""",
        unsafe_allow_html=True,
    )