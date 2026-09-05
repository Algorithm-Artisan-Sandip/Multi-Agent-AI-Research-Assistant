from unittest.mock import patch

from pipeline import run_research_pipeline
from tools import SearchHit, ScrapedPage, _dedupe, scrape_url


def test_dedupe_keeps_first_unique_url():
    hits = [
        SearchHit("A", "https://ex.com/a", "one", "tavily"),
        SearchHit("A2", "https://ex.com/a/", "dup", "ddg"),
        SearchHit("B", "https://ex.com/b", "two", "wiki"),
    ]
    unique = _dedupe(hits)
    assert [hit.url for hit in unique] == ["https://ex.com/a", "https://ex.com/b"]


def test_scrape_rejects_invalid_url():
    page = scrape_url("not-a-url")
    assert page.ok is False


def test_pipeline_answers_each_question_from_real_tools(monkeypatch):
    fake_hits = [
        SearchHit(
            title="Quantum overview",
            url="https://example.com/quantum",
            snippet="Quantum computing uses qubits for future cryptography and chemistry.",
            source="mock",
            query="Quantum computing",
            raw_content="Quantum computing uses qubits. The future includes error-corrected machines.",
        ),
        SearchHit(
            title="Challenges",
            url="https://example.com/challenges",
            snippet="Noise and decoherence are the main limitations of quantum hardware.",
            source="mock",
            query="challenges",
            raw_content="The main challenges are noise, decoherence, and expensive cryogenics.",
        ),
    ]

    def fake_search(query, max_results=5):
        return fake_hits

    def fake_wiki(query, max_results=3):
        return [
            SearchHit(
                title="Wikipedia Quantum computing",
                url="https://en.wikipedia.org/wiki/Quantum_computing",
                snippet="Quantum computing is a type of computation using quantum states.",
                source="wikipedia",
                query=query,
                raw_content="Quantum computing is a type of computation using quantum-mechanical phenomena.",
            )
        ]

    def fake_scrape(url):
        return ScrapedPage(url=url, title="page", text="fallback scraped text about quantum computing", ok=True)

    with (
        patch("agents.web_search", side_effect=fake_search),
        patch("agents.wikipedia_search", side_effect=fake_wiki),
        patch("agents.scrape_url", side_effect=fake_scrape),
        patch("llm.groq_api_key", return_value=""),
    ):
        result = run_research_pipeline("Quantum Computing Future")

    assert result["status"] == "completed"
    assert result["report"]
    assert result["search_hits"]
    assert result["answers"]
    assert len(result["answers"]) >= 4
    for item in result["answers"]:
        assert item["question"]
        assert item["answer"]
        assert "No supporting sources" not in item["answer"]
    assert "No search results available." not in (result["search_results"] or "")
    assert "No scraped content available." not in (result["scraped_content"] or "")


def test_pipeline_fails_closed_when_search_empty():
    with (
        patch("agents.web_search", return_value=[]),
        patch("agents.wikipedia_search", return_value=[]),
        patch("llm.groq_api_key", return_value=""),
    ):
        result = run_research_pipeline("Totally unknown xyzabc topic 123")

    assert result["status"] == "failed"
    assert not result.get("report")
    assert "No search results available." in (result.get("search_results") or "")
