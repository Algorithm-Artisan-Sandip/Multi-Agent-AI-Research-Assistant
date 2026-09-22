"""Clean scraped and extractive text before showing it in the UI or answers."""

from __future__ import annotations

import re


def sanitize_reading_text(text: str, max_len: int = 420) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\b(title|share|search|menu|skip to content)\s*:", " ", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"[_*`>#]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return cleaned


def sanitize_report_markdown(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            if title:
                lines.append("#" * min(hashes, 3) + " " + sanitize_reading_text(title, max_len=200))
            continue
        if line.startswith("- ") or line.startswith("* "):
            body = sanitize_reading_text(line[2:], max_len=500)
            if body:
                lines.append(f"- {body}")
            continue
        body = sanitize_reading_text(line, max_len=600)
        if body:
            lines.append(body)
    return "\n".join(lines)


def dedupe_sources(sources: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for src in sources:
        url = (src.get("url") or "").rstrip("/").lower()
        if not url or url in seen:
            continue
        seen.add(url)
        title = sanitize_reading_text(src.get("title") or url, max_len=120) or url
        unique.append({"title": title, "url": src.get("url") or url})
    return unique
