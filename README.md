# Multi-Agent AI Research Assistant

Streamlit + LangGraph research app. It boots on Streamlit Community Cloud **without** API keys, then optionally uses Groq and Tavily when they are provided as secrets.

Live app: https://multi-agent-ai-research-assistant-v1.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

1. App settings → the GitHub repo `Algorithm-Artisan-Sandip/Multi-Agent-AI-Research-Assistant`
2. Branch: **`main`**
3. Main file: **`streamlit_app.py`** (or `app.py`)
4. Optional secrets (App settings → Secrets):

```toml
GROQ_API_KEY = "your_groq_key"
TAVILY_API_KEY = "your_tavily_key"
```

Without secrets the Planner still asks questions, Search uses DuckDuckGo + Wikipedia, and Writer returns sourced extractive answers. The old `main` branch crashed on import when `.env` keys were missing; that is why the Cloud app failed to go live.
