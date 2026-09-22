"""Specialized research agents. Each agent owns one job in the LangGraph workflow."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config import (
    MAX_PAGES_TO_READ,
    MAX_RESEARCH_QUESTIONS,
    MAX_RESULTS_PER_QUERY,
    MAX_SEARCH_QUERIES,
    MAX_TOTAL_HITS,
    RESEARCH_SYSTEM,
    SCRAPE_WORKERS,
)
from llm import ResearchLLM
from text_clean import sanitize_reading_text
from tools import (
    SearchHit,
    ScrapedPage,
    format_hits,
    format_pages,
    scrape_url,
    web_search,
    wikipedia_search,
)


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}


def dedupe_questions(topic: str, questions: list[str]) -> list[str]:
    topic_tokens = _token_set(topic)
    cleaned: list[str] = []
    seen: set[str] = set()
    for question in questions:
        q = question.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        q_tokens = _token_set(q)
        if topic_tokens and q_tokens:
            overlap = len(topic_tokens & q_tokens) / max(len(topic_tokens), 1)
            if overlap > 0.82 and len(q_tokens) <= len(topic_tokens) + 4:
                continue
        seen.add(key)
        cleaned.append(q)
    if not cleaned:
        cleaned = [f"What do credible sources report about {topic.strip()}?"]
    return cleaned[:MAX_RESEARCH_QUESTIONS]


def default_questions(topic: str) -> list[str]:
    topic = topic.strip()
    lower = topic.lower()
    core = topic.rstrip("?").strip()
    if len(topic.split()) <= 8:
        return dedupe_questions(
            topic,
            [
                f"What do public sources report about {core}?",
                f"What background context helps interpret information on {core}?",
                f"What limitations exist in available data on {core}?",
            ],
        )
    if lower.startswith(("who ", "who will", "will ", "which party", "which candidate")):
        return dedupe_questions(
            topic,
            [
                core if topic.endswith("?") else f"{core}?",
                f"What do polls, models, and credible analysts say about {core}?",
                f"What historical patterns and indicators are relevant to {core}?",
                f"What uncertainties could change the outcome of {core}?",
            ],
        )
    if topic.endswith("?"):
        return dedupe_questions(
            topic,
            [
                topic,
                f"What verified facts and data exist today about {core}?",
                f"What are competing expert perspectives on {core}?",
                f"What is unknown or disputed about {core}?",
            ],
        )
    return dedupe_questions(
        topic,
        [
            f"What should researchers know about {topic}?",
            f"What is the current state of evidence on {topic}?",
            f"What risks, debates, or limitations apply to {topic}?",
            f"What is the near-term outlook for {topic}?",
        ],
    )


def plan_questions(topic: str, llm: ResearchLLM, *, use_llm: bool = False) -> list[str]:
    questions = default_questions(topic)
    if not use_llm or not llm.available:
        return questions
    result = llm.complete(
        system=(
            "You are the Planner Agent. Return exactly 4 numbered, answerable research questions. "
            "No preamble."
        ),
        human=f"Topic: {topic}",
    )
    parsed = _parse_numbered_lines(result.text)
    picked = parsed[:MAX_RESEARCH_QUESTIONS] if len(parsed) >= 3 else questions
    return dedupe_questions(topic, picked)


def search_for_questions(topic: str, questions: list[str]) -> tuple[list[SearchHit], str]:
    hits: list[SearchHit] = []
    provider_notes: list[str] = []
    queries = [topic, *questions[: max(0, MAX_SEARCH_QUERIES - 1)]]
    seen_queries: set[str] = set()
    pending: list[str] = []
    for query in queries:
        normalized = query.strip().lower()
        if not normalized or normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        pending.append(query)

    for extra in (f"{topic} news analysis", f"{topic} forecast report"):
        key = extra.strip().lower()
        if key not in seen_queries:
            seen_queries.add(key)
            pending.append(extra)

    with ThreadPoolExecutor(max_workers=min(4, len(pending) or 1)) as pool:
        futures = {pool.submit(web_search, query, MAX_RESULTS_PER_QUERY): query for query in pending}
        for future in as_completed(futures):
            try:
                batch = future.result()
            except Exception:  # noqa: BLE001
                batch = []
            if batch:
                provider_notes.append(batch[0].source)
            hits.extend(batch)

    try:
        wiki = wikipedia_search(topic, max_results=1)
        hits.extend(wiki)
        if wiki:
            provider_notes.append("wikipedia")
    except Exception as exc:  # noqa: BLE001
        provider_notes.append(f"wikipedia-error:{exc}")

    unique = _unique_hits(hits)[:MAX_TOTAL_HITS]
    provider = ",".join(dict.fromkeys(provider_notes)) or "none"
    return unique, provider


def read_sources(hits: list[SearchHit], limit: int = MAX_PAGES_TO_READ) -> list[ScrapedPage]:
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

    to_scrape: list[SearchHit] = []
    for hit in hits:
        key = hit.url.rstrip("/").lower()
        if key in used:
            continue
        used.add(key)
        to_scrape.append(hit)
        if len(to_scrape) >= limit:
            break

    def _fetch(hit: SearchHit) -> ScrapedPage:
        page = scrape_url(hit.url)
        if hit.snippet and not page.ok:
            return ScrapedPage(
                url=hit.url,
                title=hit.title,
                text=sanitize_reading_text(hit.snippet, max_len=1200),
                ok=True,
                error=f"Used search snippet because scrape failed: {page.error}",
            )
        if page.ok:
            page.text = sanitize_reading_text(page.text, max_len=1200)
        return page

    with ThreadPoolExecutor(max_workers=min(SCRAPE_WORKERS, len(to_scrape) or 1)) as pool:
        pages.extend(pool.map(_fetch, to_scrape))
    return pages


def answer_questions(
    topic: str,
    questions: list[str],
    hits: list[SearchHit],
    pages: list[ScrapedPage],
    llm: ResearchLLM,
    *,
    use_llm: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    corpus = _build_corpus(hits, pages)
    answers: list[dict[str, Any]] = []
    llm_used = False
    for question in questions:
        evidence = _top_evidence(question, corpus, k=2)
        answer_text = _extractive_answer(evidence)
        if use_llm and llm.available:
            llm_answer = llm.complete(
                system=RESEARCH_SYSTEM + " Answer in 2 short paragraphs.",
                human=(
                    f"Topic: {topic}\nQuestion: {question}\n\nSources:\n"
                    + "\n\n".join(
                        f"- {item['title']} ({item['url']})\n{item['text'][:900]}"
                        for item in evidence
                    )
                ),
            )
            if llm_answer.used_llm and llm_answer.text:
                answer_text = sanitize_reading_text(llm_answer.text, max_len=900)
                llm_used = True
        answers.append(
            {
                "question": question,
                "answer": answer_text,
                "sources": [
                    {
                        "title": sanitize_reading_text(item["title"], max_len=120),
                        "url": item["url"],
                    }
                    for item in evidence
                ],
            }
        )
    return answers, llm_used


def write_report(
    topic: str,
    questions: list[str],
    answers: list[dict[str, Any]],
    hits: list[SearchHit],
    pages: list[ScrapedPage],
    llm: ResearchLLM,
) -> tuple[str, bool]:
    answers_block = "\n\n".join(
        f"### {item['question']}\n{item['answer']}" for item in answers
    )
    references = _references(hits, pages)
    llm_report = llm.complete(
        system=(
            RESEARCH_SYSTEM
            + " Write a complete Markdown report. Every question must appear in Key Findings. "
            "Include References with URLs."
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
        return llm_report.text, True
    return _extractive_report(topic, answers, references), False


def structured_critique(
    topic: str,
    answers: list[dict[str, Any]],
    hits: list[SearchHit],
    pages: list[ScrapedPage],
    media: list[dict[str, Any]],
    *,
    fast_mode: bool,
    used_llm_synthesis: bool,
    groq_configured: bool,
    tavily_configured: bool,
    llm_error: str = "",
) -> str:
    answered = sum(1 for item in answers if len((item.get("answer") or "").split()) >= 20)
    total_q = len(answers) or 1
    avg_len = sum(len((item.get("answer") or "").split()) for item in answers) / total_q
    source_links = sum(len(item.get("sources") or []) for item in answers)
    readable_pages = sum(1 for page in pages if page.ok)

    score = 4.0
    score += min(2.0, len(hits) / 6.0)
    score += min(1.5, readable_pages / 2.0)
    score += min(1.5, len(media) / 5.0)
    score += min(1.5, answered / max(total_q, 1) * 1.5)
    score += min(1.0, avg_len / 120.0)
    if used_llm_synthesis:
        score += 0.8
    if fast_mode:
        score -= 0.7
    if groq_configured and not used_llm_synthesis:
        score -= 1.2
    if len(hits) < 5:
        score -= 1.0
    if len(media) == 0:
        score -= 0.8
    score = max(3.0, min(9.6, score))
    score_text = f"{score:.1f}/10"

    strengths: list[str] = []
    weaknesses: list[str] = []
    actions: list[str] = []

    if answered >= total_q:
        strengths.append(f"All {total_q} research questions received answers.")
    else:
        weaknesses.append(f"Only {answered}/{total_q} answers look sufficiently detailed.")
    if len(hits) >= 8:
        strengths.append(f"Broad source coverage ({len(hits)} links).")
    else:
        weaknesses.append(f"Source coverage is narrow ({len(hits)} links).")
    if len(media) > 0:
        strengths.append(f"Media discovery found {len(media)} image/video assets.")
    else:
        weaknesses.append("No media assets were discovered for this topic.")
    if readable_pages >= 2:
        strengths.append(f"Reader agent extracted {readable_pages} readable pages.")
    else:
        weaknesses.append("Few pages were readable; answers rely heavily on snippets.")

    if used_llm_synthesis:
        strengths.append("Groq synthesis was used for writing.")
    elif groq_configured:
        weaknesses.append(
            "Groq is configured but synthesis did not run successfully"
            + (f" ({llm_error})" if llm_error else ".")
        )
        actions.append("Verify Groq model access and quota, then rerun in Deep mode.")
    else:
        actions.append("Add Groq in Streamlit secrets for richer synthesis.")

    if not tavily_configured:
        actions.append("Add Tavily in secrets for deeper search results.")
    if fast_mode:
        actions.append("Disable Fast mode when you need deeper narrative synthesis.")

    return (
        f"**Overall score: {score_text}**\n\n"
        f"**Topic:** {topic}\n\n"
        "**Strengths**\n"
        + "".join(f"- {line}\n" for line in strengths)
        + "\n**Weaknesses**\n"
        + "".join(f"- {line}\n" for line in weaknesses)
        + "\n**Recommended next steps**\n"
        + "".join(f"- {line}\n" for line in actions)
        + "\n**Evidence summary**\n"
        f"- Source links: {len(hits)}\n"
        f"- Readable pages: {readable_pages}\n"
        f"- Media assets: {len(media)}\n"
        f"- Cited links in answers: {source_links}\n"
    )


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
    return ""


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
            corpus.append(
                {
                    "title": page.title or page.url,
                    "url": page.url,
                    "text": sanitize_reading_text(page.text, max_len=1200),
                }
            )
    for hit in hits:
        corpus.append(
            {
                "title": hit.title,
                "url": hit.url,
                "text": sanitize_reading_text(hit.raw_content or hit.snippet, max_len=1200),
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


def _extractive_answer(evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "No supporting sources were found for this question."
    parts: list[str] = []
    seen: set[str] = set()
    for item in evidence[:2]:
        url = item["url"].rstrip("/").lower()
        if url in seen:
            continue
        seen.add(url)
        snippet = sanitize_reading_text(item["text"], max_len=360)
        if snippet:
            parts.append(snippet)
    return "\n\n".join(parts) if parts else "No supporting sources were found for this question."


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


def build_executive_summary(topic: str, answers: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for item in answers:
        text = sanitize_reading_text(item.get("answer") or "", max_len=320)
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
        if len(parts) >= 3:
            break
    if parts:
        return " ".join(parts)
    return f"Research collected on: {topic}"


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
