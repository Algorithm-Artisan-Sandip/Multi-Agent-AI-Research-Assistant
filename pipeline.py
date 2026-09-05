"""Multi-agent research pipeline (sequential orchestration for fast Streamlit boot)."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from agents import (
    answer_questions,
    critique_report,
    plan_questions,
    read_sources,
    search_for_questions,
    write_report,
)
from llm import ResearchLLM
from tools import SearchHit, ScrapedPage, format_hits, format_pages


ProgressCb = Callable[[int, str], None]


class ResearchState(TypedDict, total=False):
    topic: str
    questions: list[str]
    search_hits: list[dict[str, Any]]
    scraped_pages: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    report: str
    feedback: str
    logs: list[str]
    errors: list[str]
    search_results: str
    scraped_content: str
    search_provider: str
    model: str
    used_llm: bool
    status: str


def _log(state: ResearchState, message: str) -> list[str]:
    logs = list(state.get("logs") or [])
    logs.append(message)
    print(message, flush=True)
    return logs


def _emit(progress: ProgressCb | None, pct: int, message: str, state: ResearchState) -> list[str]:
    if progress:
        progress(pct, message)
    return _log(state, message)


def _merge(state: ResearchState, patch: ResearchState) -> ResearchState:
    merged: ResearchState = dict(state)
    merged.update(patch)
    return merged


def _planner(state: ResearchState, llm: ResearchLLM, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 10, "Planner Agent: designing research questions", state)
    questions = plan_questions(state["topic"], llm)
    logs = _log({**state, "logs": logs}, f"Planner Agent: {len(questions)} questions")
    return _merge(state, {"questions": questions, "logs": logs, "model": llm.model_name, "used_llm": llm.available})


def _searcher(state: ResearchState, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 30, "Search Agent: gathering sources", state)
    hits, provider = search_for_questions(state["topic"], state.get("questions") or [])
    if not hits:
        errors = list(state.get("errors") or [])
        errors.append("Search Agent found no results from any provider.")
        return _merge(
            state,
            {
                "search_hits": [],
                "search_results": "No search results available.",
                "search_provider": provider,
                "errors": errors,
                "logs": _log({**state, "logs": logs}, "Search Agent: no results"),
                "status": "failed",
            },
        )
    formatted = format_hits(hits)
    logs = _log({**state, "logs": logs}, f"Search Agent: {len(hits)} unique sources ({provider})")
    return _merge(
        state,
        {
            "search_hits": [hit.to_dict() for hit in hits],
            "search_results": formatted,
            "search_provider": provider,
            "logs": logs,
        },
    )


def _reader(state: ResearchState, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 50, "Reader Agent: extracting page content", state)
    hits = [SearchHit(**item) for item in state.get("search_hits") or []]
    pages = read_sources(hits)
    ok_pages = [page for page in pages if page.ok]
    if not ok_pages and hits:
        ok_pages = [
            ScrapedPage(url=hit.url, title=hit.title, text=hit.snippet, ok=True, error="snippet-fallback")
            for hit in hits[:5]
            if hit.snippet
        ]
        pages = ok_pages
    formatted = format_pages(pages)
    logs = _log({**state, "logs": logs}, f"Reader Agent: {len(ok_pages)} readable sources")
    errors = list(state.get("errors") or [])
    if not ok_pages:
        errors.append("Reader Agent could not extract content from any source.")
        return _merge(
            state,
            {
                "scraped_pages": [page.to_dict() for page in pages],
                "scraped_content": formatted,
                "errors": errors,
                "logs": logs,
                "status": "failed",
            },
        )
    return _merge(
        state,
        {
            "scraped_pages": [page.to_dict() for page in pages],
            "scraped_content": formatted,
            "logs": logs,
        },
    )


def _writer(state: ResearchState, llm: ResearchLLM, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 75, "Writer Agent: answering each research question", state)
    hits = [SearchHit(**item) for item in state.get("search_hits") or []]
    pages = [ScrapedPage(**item) for item in state.get("scraped_pages") or []]
    questions = state.get("questions") or []
    answers = answer_questions(state["topic"], questions, hits, pages, llm)
    report = write_report(state["topic"], questions, answers, hits, pages, llm)
    if not (report or "").strip():
        errors = list(state.get("errors") or [])
        errors.append("Writer Agent produced an empty report.")
        return _merge(
            state,
            {"answers": answers, "report": "", "errors": errors, "logs": logs, "status": "failed"},
        )
    logs = _log({**state, "logs": logs}, f"Writer Agent: answered {len(answers)} questions")
    return _merge(state, {"answers": answers, "report": report, "logs": logs})


def _critic(state: ResearchState, llm: ResearchLLM, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 90, "Critic Agent: reviewing the report", state)
    feedback = critique_report(state["topic"], state.get("report") or "", state.get("answers") or [], llm)
    logs = _log({**state, "logs": logs}, "Critic Agent: review complete")
    _emit(progress, 100, "Pipeline complete", {**state, "logs": logs})
    return _merge(state, {"feedback": feedback, "logs": logs, "status": "completed", "model": llm.model_name})


def run_research_pipeline(topic: str, progress: ProgressCb | None = None) -> dict[str, Any]:
    topic = (topic or "").strip()
    if not topic:
        return {
            "status": "failed",
            "errors": ["Topic cannot be empty."],
            "search_results": "No search results available.",
            "scraped_content": "No scraped content available.",
        }

    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE STARTED")
    print("=" * 70)

    state: ResearchState = {
        "topic": topic,
        "logs": [],
        "errors": [],
        "status": "running",
        "search_results": "No search results available.",
        "scraped_content": "No scraped content available.",
    }

    try:
        llm = ResearchLLM()
        for step in (
            lambda s: _planner(s, llm, progress),
            lambda s: _searcher(s, progress),
            lambda s: _reader(s, progress),
            lambda s: _writer(s, llm, progress),
            lambda s: _critic(s, llm, progress),
        ):
            state = step(state)
            if state.get("status") == "failed":
                break
    except Exception as exc:  # noqa: BLE001
        print("Pipeline failed:", exc)
        state["status"] = "failed"
        state.setdefault("errors", []).append(str(exc))

    if state.get("status") != "failed":
        if not state.get("report"):
            state["status"] = "failed"
            state.setdefault("errors", []).append("Pipeline produced no report.")
        elif not state.get("search_hits"):
            state["status"] = "failed"
            state.setdefault("errors", []).append("Pipeline produced no search hits.")
        else:
            state["status"] = "completed"

    if state.get("status") == "completed":
        filename = topic.replace(" ", "_") + "_report.txt"
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(state.get("report") or "")
                handle.write("\n\n")
                handle.write(state.get("feedback") or "")
            print(f"Report saved as: {filename}")
        except OSError as exc:
            state.setdefault("errors", []).append(f"Could not save report file: {exc}")

    print("=" * 70)
    print(f"PIPELINE {state.get('status', 'unknown').upper()}")
    print("=" * 70)
    return dict(state)


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()
    output = run_research_pipeline(topic)
    print(output.get("status"))
    print(output.get("report", "")[:1500])
