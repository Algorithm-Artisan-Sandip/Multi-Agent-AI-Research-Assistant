"""LangGraph multi-agent research pipeline."""

from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

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


def build_graph(llm: ResearchLLM, progress: ProgressCb | None = None):
    def emit(pct: int, message: str, state: ResearchState) -> list[str]:
        if progress:
            progress(pct, message)
        return _log(state, message)

    def planner(state: ResearchState) -> ResearchState:
        topic = state["topic"]
        logs = emit(10, "Planner Agent: designing research questions", state)
        questions = plan_questions(topic, llm)
        logs = _log({**state, "logs": logs}, f"Planner Agent: {len(questions)} questions")
        return {"questions": questions, "logs": logs, "model": llm.model_name, "used_llm": llm.available}

    def searcher(state: ResearchState) -> ResearchState:
        logs = emit(30, "Search Agent: gathering sources", state)
        hits, provider = search_for_questions(state["topic"], state.get("questions") or [])
        if not hits:
            errors = list(state.get("errors") or [])
            errors.append("Search Agent found no results from any provider.")
            return {
                "search_hits": [],
                "search_results": "No search results available.",
                "search_provider": provider,
                "errors": errors,
                "logs": _log({**state, "logs": logs}, "Search Agent: no results"),
                "status": "failed",
            }
        formatted = format_hits(hits)
        logs = _log({**state, "logs": logs}, f"Search Agent: {len(hits)} unique sources ({provider})")
        return {
            "search_hits": [hit.to_dict() for hit in hits],
            "search_results": formatted,
            "search_provider": provider,
            "logs": logs,
        }

    def reader(state: ResearchState) -> ResearchState:
        logs = emit(50, "Reader Agent: extracting page content", state)
        hits = [SearchHit(**item) for item in state.get("search_hits") or []]
        pages = read_sources(hits)
        ok_pages = [page for page in pages if page.ok]
        if not ok_pages and hits:
            # Last-resort: treat snippets as readable content so later agents still answer.
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
            return {
                "scraped_pages": [page.to_dict() for page in pages],
                "scraped_content": formatted,
                "errors": errors,
                "logs": logs,
                "status": "failed",
            }
        return {
            "scraped_pages": [page.to_dict() for page in pages],
            "scraped_content": formatted,
            "logs": logs,
        }

    def writer(state: ResearchState) -> ResearchState:
        logs = emit(75, "Writer Agent: answering each research question", state)
        hits = [SearchHit(**item) for item in state.get("search_hits") or []]
        pages = [ScrapedPage(**item) for item in state.get("scraped_pages") or []]
        questions = state.get("questions") or []
        answers = answer_questions(state["topic"], questions, hits, pages, llm)
        report = write_report(state["topic"], questions, answers, hits, pages, llm)
        if not (report or "").strip():
            errors = list(state.get("errors") or [])
            errors.append("Writer Agent produced an empty report.")
            return {"answers": answers, "report": "", "errors": errors, "logs": logs, "status": "failed"}
        logs = _log({**state, "logs": logs}, f"Writer Agent: answered {len(answers)} questions")
        return {"answers": answers, "report": report, "logs": logs}

    def critic(state: ResearchState) -> ResearchState:
        logs = emit(90, "Critic Agent: reviewing the report", state)
        feedback = critique_report(state["topic"], state.get("report") or "", state.get("answers") or [], llm)
        logs = _log({**state, "logs": logs}, "Critic Agent: review complete")
        emit(100, "Pipeline complete", {**state, "logs": logs})
        return {"feedback": feedback, "logs": logs, "status": "completed", "model": llm.model_name}

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner)
    graph.add_node("searcher", searcher)
    graph.add_node("reader", reader)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    def continue_if_healthy(next_node: str):
        def _route(state: ResearchState) -> Literal["continue", "end"]:
            if state.get("status") == "failed":
                return "end"
            return "continue"

        return _route, {"continue": next_node, "end": END}

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    search_route, search_map = continue_if_healthy("reader")
    graph.add_conditional_edges("searcher", search_route, search_map)
    read_route, read_map = continue_if_healthy("writer")
    graph.add_conditional_edges("reader", read_route, read_map)
    write_route, write_map = continue_if_healthy("critic")
    graph.add_conditional_edges("writer", write_route, write_map)
    graph.add_edge("critic", END)
    return graph.compile()


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

    llm = ResearchLLM()
    app = build_graph(llm, progress=progress)
    initial: ResearchState = {
        "topic": topic,
        "logs": [],
        "errors": [],
        "status": "running",
        "search_results": "No search results available.",
        "scraped_content": "No scraped content available.",
    }
    try:
        result = app.invoke(initial)
    except Exception as exc:  # noqa: BLE001
        print("Pipeline failed:", exc)
        return {
            **initial,
            "status": "failed",
            "errors": [str(exc)],
        }

    if result.get("status") != "failed":
        if not result.get("report"):
            result["status"] = "failed"
            result.setdefault("errors", []).append("Completed graph produced no report.")
        elif not result.get("search_hits"):
            result["status"] = "failed"
            result.setdefault("errors", []).append("Completed graph produced no search hits.")
        else:
            result["status"] = "completed"

    if result.get("status") == "completed":
        filename = topic.replace(" ", "_") + "_report.txt"
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(result.get("report") or "")
                handle.write("\n\n")
                handle.write(result.get("feedback") or "")
            print(f"Report saved as: {filename}")
        except OSError as exc:
            result.setdefault("errors", []).append(f"Could not save report file: {exc}")

    print("=" * 70)
    print(f"PIPELINE {result.get('status', 'unknown').upper()}")
    print("=" * 70)
    return dict(result)


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()
    output = run_research_pipeline(topic)
    print(output.get("status"))
    print(output.get("report", "")[:1500])
