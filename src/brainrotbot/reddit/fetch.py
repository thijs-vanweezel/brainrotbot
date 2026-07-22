"""Pull candidate self/text posts from configured subreddits via Reddit RSS feeds.

RSS needs no auth and no app registration (unlike the now-gated Data API and the
403-blocked `.json` endpoint). It exposes the full self-text but not popularity
numbers, so `Story.feed_rank` (position in the already-popularity-ordered feed)
stands in for score. Parsed with stdlib `xml.etree` -- no extra dependency.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests

from ..config import Settings
from ..models import Story
from ..text.clean import html_to_text

_RSS_URL = "https://www.reddit.com/r/{sub}/{sort}/.rss"
_REQUEST_PAUSE_SECONDS = 2.0  # polite gap between feed requests
_MAX_429_RETRIES = 5          # shared/datacenter IPs (e.g. CI runners) get rate-limited harder
_RETRY_BACKOFF_BASE_SECONDS = 5.0  # used when Reddit sends no Retry-After header; doubles each retry
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
# A browser-like UA gets far fewer 403s than an API-style one.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def fetch_candidates(settings: Settings, sort: str = "top") -> list[Story]:
    headers = {"User-Agent": _BROWSER_UA, "Accept": "application/atom+xml, application/xml"}
    stories: list[Story] = []
    for i, sub in enumerate(settings.subreddits):
        if i:
            time.sleep(_REQUEST_PAUSE_SECONDS)
        params = {"t": settings.time_filter, "limit": settings.limit_per_sub}
        root = _fetch_feed(sub, sort, params, headers)
        if root is not None:
            stories.extend(_parse_feed(root, sub))
    return stories


def _fetch_feed(sub: str, sort: str, params: dict, headers: dict) -> ET.Element | None:
    """GET + parse one subreddit's feed, retrying a 429 (honoring Retry-After) before giving up."""
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            resp = requests.get(
                _RSS_URL.format(sub=sub, sort=sort), headers=headers, params=params, timeout=20
            )
            if resp.status_code == 429 and attempt < _MAX_429_RETRIES:
                default_wait = _RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt)  # 5, 10, 20, 40, 80s
                wait = float(resp.headers.get("Retry-After", default_wait))
                print(f"[brainrotbot]   r/{sub} rate-limited (429); retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{_MAX_429_RETRIES}) ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return ET.fromstring(resp.content)
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"[brainrotbot]   warning: r/{sub} feed failed ({exc}); skipping.")
            return None
    return None


def _parse_feed(root: ET.Element, sub: str) -> list[Story]:
    stories: list[Story] = []
    for rank, entry in enumerate(root.findall("a:entry", _ATOM)):
        post_id = _strip_prefix(_text(entry, "a:id"))
        if not post_id:
            continue
        content_el = entry.find("a:content", _ATOM)
        raw_body = html_to_text(content_el.text if content_el is not None else "")
        link_el = entry.find("a:link", _ATOM)
        url = link_el.get("href", "") if link_el is not None else ""
        author_el = entry.find("a:author/a:name", _ATOM)
        author = author_el.text if author_el is not None else "[unknown]"
        category_el = entry.find("a:category", _ATOM)
        flair = category_el.get("label") if category_el is not None else None

        stories.append(
            Story(
                post_id=post_id,
                subreddit=sub,
                title=_text(entry, "a:title") or "",
                raw_body=raw_body,
                url=url,
                author=author or "[unknown]",
                created_utc=_parse_ts(_text(entry, "a:published")),
                feed_rank=rank,
                flair=flair,
            )
        )
    return stories


def _text(parent: ET.Element, path: str) -> str | None:
    el = parent.find(path, _ATOM)
    return el.text if el is not None else None


def _strip_prefix(raw_id: str | None) -> str:
    """'t3_1abcd' -> '1abcd' (RSS prefixes post ids with the t3_ fullname)."""
    if not raw_id:
        return ""
    return raw_id.split("_", 1)[1] if "_" in raw_id else raw_id


def _parse_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0
