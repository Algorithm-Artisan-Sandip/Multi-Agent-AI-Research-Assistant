"""Runtime configuration. API keys are optional; tools fall back when missing."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

USER_AGENT = (
    "Mozilla/5.0 (compatible; MultiAgentResearchAssistant/2.0; "
    "+https://github.com/Algorithm-Artisan-Sandip/Multi-Agent-AI-Research-Assistant)"
)

MAX_RESEARCH_QUESTIONS = 4
MAX_SEARCH_QUERIES = 3
MAX_RESULTS_PER_QUERY = 3
MAX_TOTAL_HITS = 12
MAX_PAGES_TO_READ = 3
SCRAPE_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = 8

GROQ_MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


def _from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st

        secrets = st.secrets
        if name in secrets:
            return str(secrets[name]).strip()
        nested = secrets.get("general") if hasattr(secrets, "get") else None
        if isinstance(nested, dict) and name in nested:
            return str(nested[name]).strip()
    except Exception:
        return ""
    return ""


def _lookup(name: str) -> str:
    return (os.getenv(name) or "").strip() or _from_streamlit_secrets(name)


def groq_api_key() -> str:
    return _lookup("GROQ_API_KEY")


def tavily_api_key() -> str:
    return _lookup("TAVILY_API_KEY")


def set_runtime_keys(*, groq: str | None = None, tavily: str | None = None) -> None:
    """Inject keys for tests or local scripts (not exposed in the Streamlit UI)."""
    if groq is not None:
        os.environ["GROQ_API_KEY"] = groq.strip()
    if tavily is not None:
        os.environ["TAVILY_API_KEY"] = tavily.strip()


def integration_status() -> dict[str, bool]:
    return {"groq": bool(groq_api_key()), "tavily": bool(tavily_api_key())}


RESEARCH_SYSTEM = (
    "You are a neutral research analyst. Use only the supplied sources. "
    "The application does not block research topics: summarize public information, "
    "note uncertainty, and avoid speculation beyond the evidence. Plain text only."
)
