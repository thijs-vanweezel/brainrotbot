"""Step 6: source a random background image from Pixabay for a story's thumbnail.

Pixabay's JSON API (https://pixabay.com/api/) is free, needs only an API key (no OAuth),
requires no attribution and allows commercial use -- ideal for an automated bot. We query a
random curated search term, keep the vertical hits big enough for a 9:16 frame, pick one at
random, and download it (cached on disk by image id, like the music/video caches).

Mirrors music/ncs.py: `requests` for the JSON, stdlib `urllib` for the binary download with a
`.part` atomic rename. Resilient by design -- `search_images` returns [] on any failure so the
caller (ThumbnailMaker) can retry another term or skip the step without aborting the run.
"""

from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import requests

API_URL = "https://pixabay.com/api/"
USER_AGENT = "Mozilla/5.0 (brainrotbot)"


@dataclass
class ImageHit:
    """One Pixabay image result big enough to use as a 9:16 background."""
    id: int
    page_url: str        # pixabay.com page (provenance, logged in the ledger)
    image_url: str       # direct downloadable image (largeImageURL)
    width: int
    height: int
    tags: str = ""


def search_images(
    query: str,
    api_key: str,
    *,
    per_page: int = 50,
    min_width: int = 1080,
    min_height: int = 1920,
    orientation: str = "vertical",
) -> list[ImageHit]:
    """Return Pixabay hits for `query` meeting the minimum dimensions.

    Returns [] on any failure (no key, HTTP error, no results) so the caller stays resilient.
    `min_width`/`min_height` filter on the full-resolution size so the background can fill the
    frame without upscaling artifacts.
    """
    if not api_key:
        return []
    params = {
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "orientation": orientation,
        "safesearch": "true",
        "per_page": max(3, per_page),
        "min_width": min_width,
        "min_height": min_height,
    }
    try:
        resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except Exception:  # noqa: BLE001 -- network/HTTP/JSON drift: let the caller decide
        return []
    out: list[ImageHit] = []
    for h in hits:
        w, ht = int(h.get("imageWidth", 0)), int(h.get("imageHeight", 0))
        url = h.get("largeImageURL") or h.get("webformatURL") or ""
        if not url or w < min_width or ht < min_height:
            continue
        out.append(ImageHit(
            id=int(h.get("id", 0)),
            page_url=h.get("pageURL", ""),
            image_url=url,
            width=w, height=ht,
            tags=h.get("tags", ""),
        ))
    return out


def download_image(hit: ImageHit, cache_dir: Path) -> Path:
    """Download `hit` to `cache_dir/<id>.jpg` (cached across runs), return the local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{hit.id}.jpg"
    if out.is_file() and out.stat().st_size > 0:
        return out
    req = urllib.request.Request(hit.image_url, headers={"User-Agent": USER_AGENT})
    tmp = out.with_suffix(".jpg.part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(65536):
            f.write(chunk)
    tmp.replace(out)
    return out


def pick_term(search_terms: dict[str, list[str]], rng: random.Random | None = None) -> tuple[str, str]:
    """Pick a random (category, term) from the grouped pool.

    Two-stage (category then term within it) so each category gets equal play regardless of how
    many terms it lists; both are returned so the ledger can A/B which backdrops perform.
    """
    rng = rng or random
    cats = [c for c, terms in search_terms.items() if terms]
    if not cats:
        raise ValueError("no search terms configured")
    category = rng.choice(cats)
    return category, rng.choice(search_terms[category])


def pick_image(hits: list[ImageHit], rng: random.Random | None = None) -> ImageHit:
    """Random hit (fresh image per story; rng injectable for deterministic tests)."""
    if not hits:
        raise ValueError("no image hits to pick from")
    return (rng or random).choice(hits)
