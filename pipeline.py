"""Multi-agent research pipeline orchestrated with LangGraph."""

from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict

from agents import (
    answer_questions,
    build_executive_summary,
    critique_report,
    plan_questions,
    read_sources,
    search_for_questions,
    structured_critique,
    write_report,
)
from config import groq_api_key, tavily_api_key
from llm import ResearchLLM
from media_assets import collect_media_items
from text_clean import sanitize_report_markdown
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
    used_llm_synthesis: bool
    status: str
    fast_mode: bool
    media: list[dict[str, Any]]
    executive_summary: str
    llm_error: str


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
    questions = plan_questions(state["topic"], llm, use_llm=not state.get("fast_mode", True))
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
            for hit in hits[:3]
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
                "media": [],
                "errors": errors,
                "logs": logs,
                "status": "failed",
            },
        )
    page_dicts = [page.to_dict() for page in pages]
    media = collect_media_items(page_dicts, state.get("search_hits") or [], state.get("topic") or "")
    logs = _log({**state, "logs": logs}, f"Reader Agent: {len(media)} media assets")
    return _merge(
        state,
        {
            "scraped_pages": page_dicts,
            "scraped_content": formatted,
            "media": media,
            "logs": logs,
        },
    )


def _writer(state: ResearchState, llm: ResearchLLM, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 75, "Writer Agent: answering each research question", state)
    hits = [SearchHit(**item) for item in state.get("search_hits") or []]
    pages = [ScrapedPage(**item) for item in state.get("scraped_pages") or []]
    questions = state.get("questions") or []
    use_llm = not state.get("fast_mode", True)
    answers, answers_llm = answer_questions(
        state["topic"], questions, hits, pages, llm, use_llm=use_llm
    )
    report_llm = False
    if use_llm:
        report, report_llm = write_report(state["topic"], questions, answers, hits, pages, llm)
    else:
        from agents import _extractive_report, _references

        report = _extractive_report(state["topic"], answers, _references(hits, pages))
    used_llm_synthesis = bool(answers_llm or report_llm)
    if not (report or "").strip():
        errors = list(state.get("errors") or [])
        errors.append("Writer Agent produced an empty report.")
        return _merge(
            state,
            {"answers": answers, "report": "", "errors": errors, "logs": logs, "status": "failed"},
        )
    summary = build_executive_summary(state["topic"], answers)
    media = state.get("media") or []
    if media:
        report += "\n\n## Media & Visual Sources\n"
        for item in media[:12]:
            report += f"- ({item['type']}) {item['title']}: {item['url']}\n"
    logs = _log({**state, "logs": logs}, f"Writer Agent: answered {len(answers)} questions")
    return _merge(
        state,
        {
            "answers": answers,
            "report": report,
            "executive_summary": summary,
            "used_llm_synthesis": used_llm_synthesis,
            "llm_error": llm.error,
            "logs": logs,
        },
    )


def _critic(state: ResearchState, llm: ResearchLLM, progress: ProgressCb | None) -> ResearchState:
    logs = _emit(progress, 90, "Critic Agent: reviewing the report", state)
    hits = [SearchHit(**item) for item in state.get("search_hits") or []]
    pages = [ScrapedPage(**item) for item in state.get("scraped_pages") or []]
    feedback = ""
    if not state.get("fast_mode", True) and llm.available:
        feedback = critique_report(
            state["topic"], state.get("report") or "", state.get("answers") or [], llm
        )
    if not feedback:
        feedback = structured_critique(
            state["topic"],
            state.get("answers") or [],
            hits,
            pages,
            state.get("media") or [],
            fast_mode=bool(state.get("fast_mode", True)),
            used_llm_synthesis=bool(state.get("used_llm_synthesis")),
            groq_configured=bool(groq_api_key()),
            tavily_configured=bool(tavily_api_key()),
            llm_error=state.get("llm_error") or llm.error,
        )
    logs = _log({**state, "logs": logs}, "Critic Agent: review complete")
    _emit(progress, 100, "Pipeline complete", {**state, "logs": logs})
    return _merge(state, {"feedback": feedback, "logs": logs, "status": "completed", "model": llm.model_name})


def build_langgraph_app(llm: ResearchLLM, progress: ProgressCb | None = None):
    """Compile the LangGraph workflow (import deferred for fast Streamlit boot)."""
    from langgraph.graph import END, START, StateGraph

    def route(state: ResearchState) -> Literal["continue", "end"]:
        if state.get("status") == "failed":
            return "end"
        return "continue"

    graph = StateGraph(ResearchState)
    graph.add_node("planner", lambda state: _planner(state, llm, progress))
    graph.add_node("searcher", lambda state: _searcher(state, progress))
    graph.add_node("reader", lambda state: _reader(state, progress))
    graph.add_node("writer", lambda state: _writer(state, llm, progress))
    graph.add_node("critic", lambda state: _critic(state, llm, progress))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_conditional_edges("searcher", route, {"continue": "reader", "end": END})
    graph.add_conditional_edges("reader", route, {"continue": "writer", "end": END})
    graph.add_conditional_edges("writer", route, {"continue": "critic", "end": END})
    graph.add_edge("critic", END)
    return graph.compile()


def run_research_pipeline(
    topic: str,
    progress: ProgressCb | None = None,
    *,
    fast_mode: bool = True,
) -> dict[str, Any]:
    topic = (topic or "").strip()
    if not topic:
        return {
            "status": "failed",
            "errors": ["Topic cannot be empty."],
            "search_results": "No search results available.",
            "scraped_content": "No scraped content available.",
        }

    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE STARTED (LangGraph)")
    print("=" * 70)

    initial: ResearchState = {
        "topic": topic,
        "logs": [],
        "errors": [],
        "status": "running",
        "fast_mode": fast_mode,
        "search_results": "No search results available.",
        "scraped_content": "No scraped content available.",
        "media": [],
        "executive_summary": "",
        "used_llm_synthesis": False,
        "llm_error": "",
    }

    try:
        llm = ResearchLLM()
        app = build_langgraph_app(llm, progress=progress)
        state = app.invoke(initial)
    except Exception as exc:  # noqa: BLE001
        print("Pipeline failed:", exc)
        state = {**initial, "status": "failed", "errors": [str(exc)]}

    if state.get("status") != "failed":
        if not state.get("report"):
            state["status"] = "failed"
            state.setdefault("errors", []).append("Pipeline produced no report.")
        elif not state.get("search_hits"):
            state["status"] = "failed"
            state.setdefault("errors", []).append("Pipeline produced no search hits.")
        else:
            state["status"] = "completed"

    report = state.get("report")
    if report:
        state["report"] = sanitize_report_markdown(report)

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
