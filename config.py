"""Runtime configuration. API keys are optional; tools fall back when missing."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

USER_AGENT = (
    "Mozilla/5.0 (compatible; MultiAgentResearchAssistant/2.0; "
    "+https://github.com/Algorithm-Artisan-Sandip/Multi-Agent-AI-Research-Assistant)"
)

GROQ_MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


def groq_api_key() -> str:
    return (os.getenv("GROQ_API_KEY") or "").strip()


def tavily_api_key() -> str:
    return (os.getenv("TAVILY_API_KEY") or "").strip()


def set_runtime_keys(*, groq: str | None = None, tavily: str | None = None) -> None:
    """Allow the Streamlit UI to inject keys for the current process."""
    if groq is not None:
        os.environ["GROQ_API_KEY"] = groq.strip()
    if tavily is not None:
        os.environ["TAVILY_API_KEY"] = tavily.strip()
