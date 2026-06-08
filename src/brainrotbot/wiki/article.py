"""Fetch a random Wikipedia article's intro paragraph + category for the TikTok caption.

One MediaWiki API call does it all (verified live): `generator=random` picks a random
*main-namespace* article (`grnnamespace=0`), `prop=extracts` returns its plaintext intro
(`exintro` + `explaintext`, capped at `exchars`), and `prop=categories` returns its visible
topical categories (`clshow=!hidden` drops the maintenance/stub cruft like "Articles with
short description"). The article is intentionally unrelated to the post -- it's filler text
to pad the description and seed a category for analytics.

Mirrors the minimal HTTP pattern in `music/ncs.py` (plain `requests.get` + a custom
User-Agent + timeout). Any failure returns None; the caller then degrades the caption to
the post title rather than aborting the upload.
"""

from __future__ import annotations

import requests

USER_AGENT = "brainrotbot/1.0 (https://github.com/; TikTok caption filler)"
_CATEGORY_PREFIX = "Category:"


def _api_url(lang: str) -> str:
    return f"https://{lang or 'en'}.wikipedia.org/w/api.php"


def _parse_response(data: dict, lang: str = "en") -> dict | None:
    """Pull {title, extract, url, categories, category} out of one API JSON response.

    `categories` strips the "Category:" prefix; `category` is the first visible one (or None).
    Returns None if the response carried no usable page/extract.
    """
    pages = (data.get("query") or {}).get("pages") or {}
    for page in pages.values():
        extract = (page.get("extract") or "").strip()
        title = page.get("title") or ""
        if not extract or not title:
            continue
        cats = [
            c["title"][len(_CATEGORY_PREFIX):] if c.get("title", "").startswith(_CATEGORY_PREFIX)
            else c.get("title", "")
            for c in (page.get("categories") or [])
        ]
        cats = [c for c in cats if c]
        return {
            "title": title,
            "extract": extract,
            "url": f"https://{lang or 'en'}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "categories": cats,
            "category": cats[0] if cats else None,
        }
    return None


def fetch_random_article(settings) -> dict | None:
    """Return a random Wikipedia article's intro + category, or None on any failure.

    Reads `[wikipedia]` config (lang/extract_chars/timeout_sec). Never raises: a network or
    parse failure is logged and returns None so the caption falls back to the post title.
    """
    opts = settings.wikipedia_opts
    lang = opts.get("lang", "en")
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,          # main-namespace articles only (no Talk:/File:/Category:)
        "grnlimit": 1,
        "prop": "extracts|categories",
        "exintro": 1,               # only the lead section
        "explaintext": 1,           # plain text, no HTML/wikitext
        "exchars": int(opts.get("extract_chars", 1200)),
        "cllimit": "max",
        "clshow": "!hidden",        # drop maintenance/hidden categories
    }
    try:
        resp = requests.get(
            _api_url(lang), params=params,
            headers={"User-Agent": USER_AGENT}, timeout=float(opts.get("timeout_sec", 20)),
        )
        resp.raise_for_status()
        return _parse_response(resp.json(), lang)
    except Exception as exc:  # noqa: BLE001 -- a caption is best-effort; never block the upload
        print(f"[brainrotbot]   (Wikipedia fetch failed, caption falls back to title: {exc})")
        return None
