"""Search and scrape tools used by the research agents.

Search tries Tavily first (when a key is present), then DuckDuckGo, then Wikipedia.
Scraping is best-effort and never raises to the agents.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from config import USER_AGENT, tavily_api_key

REQUEST_TIMEOUT = 12
MAX_SCRAPE_CHARS = 6000
MAX_SNIPPET_CHARS = 1200


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str
    query: str = ""
    raw_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    ok: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/json"})
    return session


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _valid_http_url(url: str) -> bool:
    parsed = urlparse(url or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _tavily_search(query: str, max_results: int) -> list[SearchHit]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=tavily_api_key())
    payload = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=True,
    )
    hits: list[SearchHit] = []
    answer = (payload.get("answer") or "").strip()
    for item in payload.get("results") or []:
        url = item.get("url") or ""
        if not _valid_http_url(url):
            continue
        snippet = _clean_text(item.get("content") or "")
        if answer and not snippet:
            snippet = answer
        hits.append(
            SearchHit(
                title=item.get("title") or url,
                url=url,
                snippet=snippet[:MAX_SNIPPET_CHARS],
                source="tavily",
                query=query,
                raw_content=_clean_text(item.get("raw_content") or "")[:MAX_SCRAPE_CHARS],
            )
        )
    if answer and hits:
        hits[0].snippet = f"{answer}\n\n{hits[0].snippet}"[:MAX_SNIPPET_CHARS]
    return hits


def _parse_html(markup: str):
    try:
        return BeautifulSoup(markup, "lxml")
    except Exception:
        return BeautifulSoup(markup, "html.parser")


def _ddg_instant(query: str) -> list[SearchHit]:
    response = _session().get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    hits: list[SearchHit] = []

    abstract_url = data.get("AbstractURL") or ""
    abstract = _clean_text(data.get("AbstractText") or "")
    if _valid_http_url(abstract_url) and abstract:
        hits.append(
            SearchHit(
                title=data.get("Heading") or query,
                url=abstract_url,
                snippet=abstract[:MAX_SNIPPET_CHARS],
                source="duckduckgo-instant",
                query=query,
                raw_content=abstract[:MAX_SCRAPE_CHARS],
            )
        )

    def _consume(topic: dict) -> None:
        url = topic.get("FirstURL") or ""
        text = _clean_text(topic.get("Text") or "")
        if _valid_http_url(url) and text:
            hits.append(
                SearchHit(
                    title=text.split(" - ", 1)[0][:120],
                    url=url,
                    snippet=text[:MAX_SNIPPET_CHARS],
                    source="duckduckgo-instant",
                    query=query,
                )
            )
        for nested in topic.get("Topics") or []:
            _consume(nested)

    for topic in data.get("RelatedTopics") or []:
        _consume(topic)
    return hits


def _ddg_search(query: str, max_results: int) -> list[SearchHit]:
    from ddgs import DDGS

    hits: list[SearchHit] = []
    with DDGS() as client:
        rows = list(client.text(query, max_results=max_results))
    for row in rows:
        url = row.get("href") or row.get("url") or ""
        if not _valid_http_url(url):
            continue
        hits.append(
            SearchHit(
                title=row.get("title") or url,
                url=url,
                snippet=_clean_text(row.get("body") or row.get("snippet") or "")[:MAX_SNIPPET_CHARS],
                source="duckduckgo",
                query=query,
            )
        )
    return hits


def wikipedia_search(query: str, max_results: int = 3) -> list[SearchHit]:
    session = _session()
    search_resp = session.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
            "utf8": 1,
        },
        timeout=REQUEST_TIMEOUT,
    )
    search_resp.raise_for_status()
    titles = [row["title"] for row in search_resp.json().get("query", {}).get("search", [])]
    if not titles:
        return []

    extract_resp = session.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "redirects": 1,
            "format": "json",
            "titles": "|".join(titles),
        },
        timeout=REQUEST_TIMEOUT,
    )
    extract_resp.raise_for_status()
    pages = extract_resp.json().get("query", {}).get("pages", {})
    hits: list[SearchHit] = []
    for page in pages.values():
        if page.get("missing") is not None:
            continue
        url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(page.get('title', ''))}"
        extract = _clean_text(page.get("extract") or "")
        hits.append(
            SearchHit(
                title=page.get("title") or url,
                url=url,
                snippet=extract[:MAX_SNIPPET_CHARS],
                source="wikipedia",
                query=query,
                raw_content=extract[:MAX_SCRAPE_CHARS],
            )
        )
    return hits


def web_search(query: str, max_results: int = 5) -> list[SearchHit]:
    """Search the web. Never raises; returns whatever providers succeed."""
    errors: list[str] = []
    hits: list[SearchHit] = []

    if tavily_api_key():
        try:
            hits.extend(_tavily_search(query, max_results))
        except Exception as exc:  # noqa: BLE001 — provider failures must not kill the pipeline
            errors.append(f"tavily: {exc}")

    if len(hits) < max_results:
        try:
            ddg_hits = _ddg_search(query, max_results)
            hits.extend(_dedupe(hits + ddg_hits)[len(hits) :])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"duckduckgo: {exc}")

    if len(hits) < max_results:
        try:
            hits.extend(_dedupe(hits + _ddg_instant(query))[len(hits) :])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"duckduckgo-instant: {exc}")

    # Always include Wikipedia so Streamlit Cloud still has sources if DDG is blocked.
    try:
        wiki_hits = wikipedia_search(query, max_results=max(3, max_results))
        hits.extend(_dedupe(hits + wiki_hits)[len(hits) :])
    except Exception as extra:  # noqa: BLE001
        errors.append(f"wikipedia: {extra}")

    if not hits and errors:
        raise RuntimeError("All search providers failed: " + "; ".join(errors))

    return _dedupe(hits)[: max(max_results, 5)]


def _dedupe(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    unique: list[SearchHit] = []
    for hit in hits:
        key = (hit.url or "").rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def scrape_url(url: str) -> ScrapedPage:
    if not _valid_http_url(url):
        return ScrapedPage(url=url, title="", text="", ok=False, error="Invalid URL")

    try:
        response = _session().get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return ScrapedPage(url=url, title="", text="", ok=False, error="PDF skipped")

        soup = _parse_html(response.text)
        title = _clean_text(soup.title.get_text() if soup.title else url)
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript", "form"]):
            tag.decompose()
        text = _clean_text(soup.get_text(separator=" ", strip=True))[:MAX_SCRAPE_CHARS]
        if len(text) < 80:
            return ScrapedPage(url=url, title=title, text=text, ok=False, error="Page had too little text")
        return ScrapedPage(url=url, title=title, text=text, ok=True)
    except Exception as exc:  # noqa: BLE001
        return ScrapedPage(url=url, title="", text="", ok=False, error=str(exc))


def scrape_urls(urls: list[str], limit: int = 5) -> list[ScrapedPage]:
    pages: list[ScrapedPage] = []
    for url in urls[:limit]:
        pages.append(scrape_url(url))
    return pages


def format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No search results available."
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"{i}. {hit.title}\n   URL: {hit.url}\n   Source: {hit.source}\n   Query: {hit.query}\n   {hit.snippet}"
        )
    return "\n\n".join(blocks)


def format_pages(pages: list[ScrapedPage]) -> str:
    if not pages:
        return "No scraped content available."
    blocks = []
    for i, page in enumerate(pages, start=1):
        if page.ok:
            blocks.append(f"{i}. {page.title}\n   URL: {page.url}\n   {page.text[:2000]}")
        else:
            blocks.append(f"{i}. FAILED {page.url}\n   {page.error}")
    return "\n\n".join(blocks)


# LangChain tool wrappers (optional ReAct agents / notebooks)
try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    tool = None  # type: ignore[assignment]
else:

    @tool("web_search")
    def web_search_tool(query: str) -> str:
        """Search the web and return titled results with URLs and snippets."""
        try:
            return format_hits(web_search(query))
        except Exception as exc:  # noqa: BLE001
            return f"Search error: {exc}"

    @tool("scrape_url")
    def scrape_url_tool(url: str) -> str:
        """Fetch a webpage and return cleaned readable text."""
        page = scrape_url(url)
        if not page.ok:
            return f"Scraping error for {url}: {page.error}"
        return f"{page.title}\n{page.url}\n{page.text}"
