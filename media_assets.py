"""Collect images and video links from pages and image search."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

_VIDEO_HOST = re.compile(r"(youtube\.com|youtu\.be|vimeo\.com|dailymotion\.com)", re.I)


def _abs_url(base: str, link: str) -> str:
    if not link:
        return ""
    link = link.strip()
    if link.startswith("//"):
        return "https:" + link
    if link.startswith("http"):
        return link
    return urljoin(base, link)


def extract_media_from_html(html: str, page_url: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    images: list[str] = []
    videos: list[str] = []
    seen_i: set[str] = set()
    seen_v: set[str] = set()

    def add_image(url: str) -> None:
        key = url.lower()
        if url and key not in seen_i:
            seen_i.add(key)
            images.append(url)

    def add_video(url: str) -> None:
        key = url.lower()
        if url and key not in seen_v:
            seen_v.add(key)
            videos.append(url)

    for prop in ("og:image", "twitter:image", "og:image:url", "og:image:secure_url"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            add_image(_abs_url(page_url, tag["content"]))

    for prop in ("og:video", "og:video:url", "og:video:secure_url"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            add_video(_abs_url(page_url, tag["content"]))

    for video_tag in soup.find_all("video"):
        src = video_tag.get("src") or ""
        if src:
            add_video(_abs_url(page_url, src))
        for source in video_tag.find_all("source", src=True):
            add_video(_abs_url(page_url, source["src"]))

    for iframe in soup.find_all("iframe", src=True):
        src = _abs_url(page_url, iframe["src"])
        if _VIDEO_HOST.search(src):
            add_video(src)

    for anchor in soup.find_all("a", href=True):
        href = _abs_url(page_url, anchor["href"])
        if _VIDEO_HOST.search(href):
            add_video(href)

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src"):
            src = img.get(attr)
            if src:
                add_image(_abs_url(page_url, src))
                break

    return images[:16], videos[:8]


def image_search_media(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        from ddgs import DDGS

        with DDGS() as client:
            rows = list(client.images(query, max_results=max_results))
        for row in rows:
            url = row.get("image") or row.get("thumbnail") or ""
            if not url.startswith("http"):
                continue
            items.append(
                {
                    "type": "image",
                    "url": url,
                    "title": (row.get("title") or "Image result")[:120],
                    "source_page": row.get("url") or "",
                    "origin": "image-search",
                }
            )
    except Exception:
        return []
    return items


def video_search_media(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        from ddgs import DDGS

        with DDGS() as client:
            rows = list(client.videos(query, max_results=max_results))
        for row in rows:
            url = row.get("content") or row.get("embed_url") or row.get("url") or ""
            if not url.startswith("http"):
                continue
            items.append(
                {
                    "type": "video",
                    "url": url,
                    "title": (row.get("title") or "Video result")[:120],
                    "source_page": row.get("url") or url,
                    "origin": "video-search",
                }
            )
    except Exception:
        return []
    return items


def collect_media_items(
    pages: list[dict],
    hits: list[dict],
    topic: str = "",
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        url = (item.get("url") or "").rstrip("/").lower()
        if not url or url in seen:
            return
        seen.add(url)
        items.append(item)

    for page in pages:
        if not page.get("ok"):
            continue
        title = page.get("title") or page.get("url") or "Source"
        src = page.get("url") or ""
        for url in page.get("image_urls") or ([page["image_url"]] if page.get("image_url") else []):
            add({"type": "image", "url": url, "title": title, "source_page": src, "origin": "page"})
        for url in page.get("video_urls") or ([page["video_url"]] if page.get("video_url") else []):
            add({"type": "video", "url": url, "title": title, "source_page": src, "origin": "page"})

    for hit in hits:
        title = hit.get("title") or hit.get("url") or "Source"
        url = hit.get("url") or ""
        blob = f"{hit.get('snippet') or ''} {hit.get('raw_content') or ''}"
        for match in re.findall(r"https?://[^\s\"']+", blob):
            cleaned = match.rstrip(".,)")
            if any(x in cleaned.lower() for x in (".jpg", ".jpeg", ".png", ".webp", ".gif")):
                add(
                    {
                        "type": "image",
                        "url": cleaned,
                        "title": title,
                        "source_page": url,
                        "origin": "snippet",
                    }
                )

    if topic.strip():
        for row in image_search_media(topic, max_results=12):
            add(row)
        for row in video_search_media(topic, max_results=6):
            add(row)

    return items[:36]
