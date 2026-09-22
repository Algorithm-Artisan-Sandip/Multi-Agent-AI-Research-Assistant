"""Collect images and video links from scraped pages for the report UI."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

_IMAGE_EXT = re.compile(r"\.(jpg|jpeg|png|webp|gif)(\?|$)", re.I)
_VIDEO_HOST = re.compile(r"(youtube\.com|youtu\.be|vimeo\.com)", re.I)


def _abs_url(base: str, link: str) -> str:
    if not link:
        return ""
    link = link.strip()
    if link.startswith("//"):
        return "https:" + link
    if link.startswith("http"):
        return link
    return urljoin(base, link)


def extract_media_from_html(html: str, page_url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    image = ""
    video = ""
    for prop in ("og:image", "twitter:image", "og:image:url"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            image = _abs_url(page_url, tag["content"])
            break
    for prop in ("og:video", "og:video:url", "og:video:secure_url"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            video = _abs_url(page_url, tag["content"])
            break
    if not video:
        iframe = soup.find("iframe", src=True)
        if iframe and _VIDEO_HOST.search(iframe["src"] or ""):
            video = _abs_url(page_url, iframe["src"])
    if not image:
        for img in soup.find_all("img", src=True):
            src = _abs_url(page_url, img["src"])
            if src and _IMAGE_EXT.search(src):
                image = src
                break
    return image, video


def collect_media_items(pages: list[dict], hits: list[dict]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, url: str, title: str, source_url: str, *, trusted: bool = False) -> None:
        key = url.rstrip("/").lower()
        if not url or key in seen:
            return
        if kind == "image" and not trusted:
            if not (
                _IMAGE_EXT.search(url)
                or any(x in url.lower() for x in ("image", "thumb", "photo", "media", "cdn"))
            ):
                return
        seen.add(key)
        items.append(
            {
                "type": kind,
                "url": url,
                "title": title[:120],
                "source_page": source_url,
            }
        )

    for page in pages:
        if not page.get("ok"):
            continue
        title = page.get("title") or page.get("url") or "Source"
        src = page.get("url") or ""
        if page.get("image_url"):
            add("image", page["image_url"], title, src, trusted=True)
        if page.get("video_url"):
            add("video", page["video_url"], title, src, trusted=True)

    for hit in hits:
        url = hit.get("url") or ""
        title = hit.get("title") or url
        # Some providers embed image URLs in raw content — best-effort only
        for match in re.findall(r"https?://\S+\.(?:jpg|jpeg|png|webp|gif)(?:\?\S*)?", hit.get("snippet") or "", re.I):
            add("image", match.rstrip(".,)"), title, url)

    return items[:24]
