# Multi-Agent AI Research Assistant

Streamlit multi-agent research app. It boots quickly on Streamlit Community Cloud **without** API keys, then optionally uses Groq and Tavily from secrets.

Live app: https://multi-agent-ai-research-assistant-v1.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud settings

| Setting | Value |
|---------|--------|
| Branch | `main` |
| Main file | `streamlit_app.py` |
| Sharing | **Public** (required or visitors see a login wall) |

Optional secrets:

```toml
GROQ_API_KEY = "your_groq_key"
TAVILY_API_KEY = "your_tavily_key"
```

Heavy agent code loads only when you click **Generate Research Report**, so cold starts stay fast.
