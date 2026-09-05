"""Specialized research agents. Each agent owns one job in the LangGraph workflow."""

from __future__ import annotations

import re
from typing import Any

from llm import ResearchLLM
from tools import (
    SearchHit,
    ScrapedPage,
    format_hits,
    format_pages,
    scrape_url,
    web_search,
    wikipedia_search,
)


def default_questions(topic: str) -> list[str]:
    topic = topic.strip()
    if topic.endswith("?"):
        core = topic.rstrip("?").strip()
        return [
            topic if topic.endswith("?") else f"{core}?",
            f"What is the current evidence and state of the art for {core}?",
            f"What challenges, risks, or limitations relate to {core}?",
            f"What is the likely outlook for {core} over the next 5–10 years?",
            f"Which organizations, papers, or products matter most for {core}?",
        ]
    return [
        f"Give a clear overview of {topic} and why it matters.",
        f"What is the current state of {topic}?",
        f"What are the main challenges, risks, or limitations around {topic}?",
        f"What is the likely future of {topic} over the next 5–10 years?",
        f"Which organizations, papers, or products are most important for {topic}?",
    ]


def plan_questions(topic: str, llm: ResearchLLM) -> list[str]:
    result = llm.complete(
        system=(
            "You are the Planner Agent for a multi-agent research assistant. "
            "Produce 5 specific, answerable research questions about the topic. "
            "Return ONLY a numbered list, no preamble."
        ),
        human=f"Topic: {topic}",
    )
    questions = _parse_numbered_lines(result.text)
    return questions[:5] if len(questions) >= 3 else default_questions(topic)


def search_for_questions(topic: str, questions: list[str]) -> tuple[list[SearchHit], str]:
    hits: list[SearchHit] = []
    provider_notes: list[str] = []
    queries = [topic, *questions]
    seen_queries: set[str] = set()
    for query in queries:
        normalized = query.strip().lower()
        if not normalized or normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        batch = web_search(query, max_results=5)
        if batch:
            provider_notes.append(batch[0].source)
        hits.extend(batch)

    try:
        wiki = wikipedia_search(topic, max_results=2)
        hits.extend(wiki)
        if wiki:
            provider_notes.append("wikipedia")
    except Exception as exc:  # noqa: BLE001
        provider_notes.append(f"wikipedia-error:{exc}")

    unique = _unique_hits(hits)
    provider = ",".join(dict.fromkeys(provider_notes)) or "none"
    return unique, provider


def read_sources(hits: list[SearchHit], limit: int = 5) -> list[ScrapedPage]:
    pages: list[ScrapedPage] = []
    used: set[str] = set()

    for hit in hits:
        if hit.raw_content and len(hit.raw_content) > 200:
            key = hit.url.rstrip("/").lower()
            if key in used:
                continue
            used.add(key)
            pages.append(
                ScrapedPage(url=hit.url, title=hit.title, text=hit.raw_content, ok=True)
            )
        if len(pages) >= limit:
            return pages

    for hit in hits:
        key = hit.url.rstrip("/").lower()
        if key in used:
            continue
        used.add(key)
        page = scrape_url(hit.url)
        if hit.snippet and not page.ok:
            page = ScrapedPage(
                url=hit.url,
                title=hit.title,
                text=hit.snippet,
                ok=True,
                error=f"Used search snippet because scrape failed: {page.error}",
            )
        pages.append(page)
        if sum(1 for item in pages if item.ok) >= limit:
            break
    return pages


def answer_questions(
    topic: str,
    questions: list[str],
    hits: list[SearchHit],
    pages: list[ScrapedPage],
    llm: ResearchLLM,
) -> list[dict[str, Any]]:
    corpus = _build_corpus(hits, pages)
    answers: list[dict[str, Any]] = []
    for question in questions:
        evidence = _top_evidence(question, corpus, k=4)
        llm_answer = llm.complete(
            system=(
                "You are the Analyst Agent. Answer the research question using ONLY the "
                "provided sources. Cite URLs inline. If sources are weak, say what is known "
                "and what is missing. 2–4 short paragraphs."
            ),
            human=(
                f"Topic: {topic}\nQuestion: {question}\n\nSources:\n"
                + "\n\n".join(
                    f"- {item['title']} ({item['url']})\n{item['text'][:1500]}"
                    for item in evidence
                )
            ),
        )
        answer_text = llm_answer.text if llm_answer.used_llm else _extractive_answer(question, evidence)
        answers.append(
            {
                "question": question,
                "answer": answer_text,
                "sources": [{"title": item["title"], "url": item["url"]} for item in evidence],
            }
        )
    return answers


def write_report(
    topic: str,
    questions: list[str],
    answers: list[dict[str, Any]],
    hits: list[SearchHit],
    pages: list[ScrapedPage],
    llm: ResearchLLM,
) -> str:
    answers_block = "\n\n".join(
        f"### {item['question']}\n{item['answer']}" for item in answers
    )
    references = _references(hits, pages)
    llm_report = llm.complete(
        system=(
            "You are the Writer Agent. Write a complete research report in Markdown. "
            "Every research question must be answered in Key Findings. Use only the "
            "provided research. Include a References section with real URLs."
        ),
        human=(
            f"Topic: {topic}\n\nResearch Q&A:\n{answers_block}\n\n"
            f"Search notes:\n{format_hits(hits)[:4000]}\n\n"
            f"Scraped notes:\n{format_pages(pages)[:4000]}\n\n"
            "Structure:\n# Introduction\n# Background\n# Key Findings\n"
            "# Challenges and Risks\n# Outlook\n# Conclusion\n# References"
        ),
    )
    if llm_report.used_llm and llm_report.text:
        return llm_report.text
    return _extractive_report(topic, answers, references)


def critique_report(topic: str, report: str, answers: list[dict[str, Any]], llm: ResearchLLM) -> str:
    llm_review = llm.complete(
        system=(
            "You are the Critic Agent. Score the report 1–10 and list strengths, "
            "weaknesses, and missing evidence. Be strict if sources are thin."
        ),
        human=f"Topic: {topic}\nQuestions answered: {len(answers)}\n\n{report}",
    )
    if llm_review.used_llm and llm_review.text:
        return llm_review.text
    source_count = sum(len(item.get("sources") or []) for item in answers)
    answered = sum(1 for item in answers if (item.get("answer") or "").strip())
    score = min(10, 4 + answered + min(3, source_count // 3))
    return (
        f"Score: {score}/10\n\n"
        "Strengths\n"
        f"- Answered {answered}/{len(answers) or 0} research questions with sourced snippets.\n"
        f"- Gathered {source_count} supporting source links.\n\n"
        "Weaknesses\n"
        "- Groq LLM was not available, so the report is extractive rather than synthesized.\n"
        "- Some live pages may have failed to scrape; snippets were used instead.\n\n"
        "Suggestions\n"
        "- Add GROQ_API_KEY for a richer Writer/Critic pass.\n"
        "- Add TAVILY_API_KEY for deeper search results.\n\n"
        "Overall Verdict\n"
        "The pipeline returned real sources and a per-question answer instead of an empty success state."
    )


def _parse_numbered_lines(text: str) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _unique_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    unique: list[SearchHit] = []
    for hit in hits:
        key = hit.url.rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def _build_corpus(hits: list[SearchHit], pages: list[ScrapedPage]) -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    for page in pages:
        if page.ok and page.text:
            corpus.append({"title": page.title or page.url, "url": page.url, "text": page.text})
    for hit in hits:
        corpus.append(
            {
                "title": hit.title,
                "url": hit.url,
                "text": hit.raw_content or hit.snippet,
            }
        )
    return corpus


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}


def _top_evidence(question: str, corpus: list[dict[str, str]], k: int = 4) -> list[dict[str, str]]:
    question_tokens = _tokenize(question)
    scored: list[tuple[int, dict[str, str]]] = []
    for item in corpus:
        tokens = _tokenize(item["text"] + " " + item["title"])
        score = len(question_tokens & tokens)
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    picked: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, item in scored:
        key = item["url"].rstrip("/").lower()
        if key in seen or not item.get("text"):
            continue
        seen.add(key)
        picked.append(item)
        if len(picked) >= k:
            break
    if not picked:
        picked = corpus[:k]
    return picked


def _extractive_answer(question: str, evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "No supporting sources were found for this question."
    parts = [f"**{question}**\n"]
    for item in evidence[:3]:
        snippet = item["text"][:500].rstrip()
        parts.append(f"- From [{item['title']}]({item['url']}): {snippet}")
    return "\n".join(parts)


def _references(hits: list[SearchHit], pages: list[ScrapedPage]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in [*pages, *hits]:
        url = getattr(item, "url", "")
        title = getattr(item, "title", "") or url
        key = url.rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        refs.append((title, url))
    return refs


def _extractive_report(topic: str, answers: list[dict[str, Any]], references: list[tuple[str, str]]) -> str:
    findings = "\n\n".join(f"### {item['question']}\n\n{item['answer']}" for item in answers)
    refs = "\n".join(f"- [{title}]({url})" for title, url in references)
    return f"""# Research Report: {topic}

# Introduction

This report was produced by a multi-agent research pipeline (Planner → Search → Reader → Writer → Critic).
Each research question below is answered from live search and page evidence.

# Background

The agents collected web results and readable source text for **{topic}**, then assembled
per-question answers so the UI cannot report success with empty research.

# Key Findings

{findings}

# Challenges and Risks

Evidence quality depends on which pages could be fetched. Failed scrapes fall back to search snippets.

# Outlook

See the forward-looking question in Key Findings for the best-supported view of what comes next.

# Conclusion

The pipeline returned sourced answers for each planned research question about {topic}.

# References

{refs or '- No URLs collected.'}
"""
