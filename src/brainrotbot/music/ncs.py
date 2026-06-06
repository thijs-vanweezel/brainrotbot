"""Step 5: discover + download NCS background-music tracks (instrumental only).

Per run we scrape ncs.io's instrumental-versions listing once (cached to disk with a TTL)
and pick one track at random per story. The chosen instrumental MP3 is downloaded to the
cache (keyed by track UUID, so picks across runs accumulate) and handed to compose() for
the soft-bed mix.

NCS isn't bot-aggressive -- no retry/backoff loop, no rotating user-agents, no rate-limit
sleeps. A plain `requests.get` + BeautifulSoup pass is enough; the TTL cache exists only to
avoid the few redundant page fetches, not to dodge throttling.

What ncs.io exposes per track (parsed from the data-* attrs on each .player-play <a>):
    data-tid          UUID -- the download endpoint key
    data-track        Title
    data-artistraw    Artist (clean, no nested HTML)
    data-genre        Genre name (one)
    data-versions     CSV; we drop tracks that don't list "Instrumental"
Mood tags live in the same <tr> as <a class="tag" href="/music-search?mood=N"> elements.
The download URL pattern is stable:
    Regular (vocals):       /track/download/<tid>
    Instrumental (no vocals)/track/download/i_<tid>     <-- we always grab this one
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://ncs.io"
LISTING = f"{BASE}/music-search"
USER_AGENT = "Mozilla/5.0 (brainrotbot)"
_CATALOGUE_FILE = "catalogue.json"


@dataclass
class TrackMeta:
    """One NCS track that has an instrumental cut available."""
    track_id: str
    title: str
    artist: str
    genre: str
    moods: list[str] = field(default_factory=list)
    page_url: str = ""               # /<slug> on ncs.io (track detail page)
    instrumental_url: str = ""       # full /track/download/i_<tid> URL

    @classmethod
    def from_dict(cls, d: dict) -> "TrackMeta":
        return cls(**d)


def _parse_listing(html: str) -> list[TrackMeta]:
    """Pull all instrumental-having tracks out of one /music-search results page."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[TrackMeta] = []
    # Each track row carries a .player-play anchor with all the metadata as data-* attrs.
    for play in soup.select("a.player-play"):
        versions = (play.get("data-versions") or "").lower()
        if "instrumental" not in versions:
            continue   # vocals-only track, skip
        tid = play.get("data-tid") or ""
        if not tid:
            continue
        # Mood tags: <a class="tag" href="/music-search?mood=N">label</a>; one row may have many.
        # Search up the DOM for the surrounding <tr> so we don't pick up moods from other rows.
        row = play.find_parent("tr")
        moods: list[str] = []
        page_url = ""
        if row is not None:
            for tag in row.select("a.tag[href*='mood=']"):
                label = tag.get_text(strip=True)
                if label:
                    moods.append(label)
            # The track-detail link is the first non-tag <a href> in the row (e.g. /c_pullmedown).
            for a in row.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/") and "music-search" not in href and "panel" not in (a.get("class") or []):
                    page_url = BASE + href
                    break
        out.append(TrackMeta(
            track_id=tid,
            title=play.get("data-track") or "",
            artist=play.get("data-artistraw") or "",
            genre=play.get("data-genre") or "",
            moods=moods,
            page_url=page_url,
            instrumental_url=f"{BASE}/track/download/i_{tid}",
        ))
    return out


def _fetch_pages(num_pages: int) -> list[TrackMeta]:
    """Scrape the first `num_pages` of the instrumental listing. NCS uses ?page=N (1-indexed)."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    tracks: list[TrackMeta] = []
    seen: set[str] = set()
    for page in range(1, max(1, num_pages) + 1):
        params = {"q": "", "genre": "", "mood": "", "version": "Instrumental", "page": page}
        resp = session.get(LISTING, params=params, timeout=30)
        resp.raise_for_status()
        page_tracks = _parse_listing(resp.text)
        if not page_tracks:
            break   # past the last populated page
        for t in page_tracks:
            if t.track_id in seen:
                continue
            seen.add(t.track_id)
            tracks.append(t)
    return tracks


def discover_instrumental_tracks(
    cache_dir: Path,
    *,
    ttl_days: int = 7,
    num_pages: int = 3,
) -> list[TrackMeta]:
    """Return the catalogue of instrumental-having NCS tracks, scraping if cache is stale.

    Catalogue is JSON in `cache_dir/catalogue.json` with a `fetched_at` epoch. If the scrape
    fails (network down, NCS HTML drift) and a stale catalogue exists we reuse it -- a slightly
    old pool is far better than aborting Step 5 for the whole run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cat_path = cache_dir / _CATALOGUE_FILE
    if cat_path.is_file():
        try:
            data = json.loads(cat_path.read_text(encoding="utf-8"))
            fresh = (time.time() - data.get("fetched_at", 0)) < ttl_days * 86400
            if fresh and data.get("tracks"):
                return [TrackMeta.from_dict(t) for t in data["tracks"]]
        except (OSError, ValueError, json.JSONDecodeError):
            pass   # corrupted cache -- fall through to a fresh scrape
    try:
        tracks = _fetch_pages(num_pages)
    except Exception:
        # Network / HTTP failure: prefer a stale catalogue over an empty pool.
        if cat_path.is_file():
            data = json.loads(cat_path.read_text(encoding="utf-8"))
            return [TrackMeta.from_dict(t) for t in data.get("tracks", [])]
        raise
    cat_path.write_text(
        json.dumps({"fetched_at": time.time(), "tracks": [asdict(t) for t in tracks]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return tracks


def download_track(track: TrackMeta, cache_dir: Path) -> Path:
    """Download `track`'s instrumental MP3 to `cache_dir/<track_id>.mp3` (cached across runs)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{track.track_id}.mp3"
    if out.is_file() and out.stat().st_size > 0:
        return out
    req = urllib.request.Request(track.instrumental_url, headers={"User-Agent": USER_AGENT})
    tmp = out.with_suffix(".mp3.part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(65536):
            f.write(chunk)
    tmp.replace(out)
    return out


def pick_track(tracks: list[TrackMeta], rng: random.Random | None = None) -> TrackMeta:
    """Random track from the catalogue (one per story; rng injectable for deterministic tests)."""
    if not tracks:
        raise ValueError("track catalogue is empty")
    return (rng or random).choice(tracks)
