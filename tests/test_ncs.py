"""Tests for the NCS scraper. The HTML fixture is a real /music-search?version=Instrumental
response captured once; if NCS redesigns the listing the fixture needs refreshing."""
from pathlib import Path

import pytest

from brainrotbot.music.ncs import (
    TrackMeta, _parse_listing, discover_instrumental_tracks, pick_track,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ncs_instrumental_page.html"


def _html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_parse_listing_returns_tracks_with_instrumental_metadata():
    tracks = _parse_listing(_html())
    # The fixture page exposes ~20 instrumental-having tracks.
    assert len(tracks) >= 10
    for t in tracks:
        assert isinstance(t, TrackMeta)
        assert t.track_id and "-" in t.track_id  # UUID
        assert t.title and t.artist
        # Download URL points to the no-vocals cut (i_ prefix), never the regular one.
        assert t.instrumental_url == f"https://ncs.io/track/download/i_{t.track_id}"
        assert t.genre  # one-per-track in data-genre


def test_parse_listing_drops_vocals_only_tracks(monkeypatch):
    """A row whose data-versions doesn't list 'Instrumental' must be filtered out."""
    html = (
        '<tr><td><a class="player-play" data-tid="x-vocals" data-track="V" '
        'data-artistraw="A" data-genre="Pop" data-versions="Regular"></a></td></tr>'
        '<tr><td><a class="player-play" data-tid="x-instr" data-track="I" '
        'data-artistraw="A" data-genre="Pop" data-versions="Regular, Instrumental"></a></td></tr>'
    )
    tracks = _parse_listing(html)
    tids = [t.track_id for t in tracks]
    assert "x-vocals" not in tids
    assert "x-instr" in tids


def test_parse_listing_extracts_mood_tags():
    """Mood tags are <a class="tag" href=".../music-search?mood=N">label</a>; genre links are
    on the same row and must not leak into moods."""
    tracks = _parse_listing(_html())
    # At least one track should have moods recorded (the fixture's first row does).
    with_moods = [t for t in tracks if t.moods]
    assert with_moods, "expected at least one track with mood tags"
    # The first track in the fixture is CERES/TAME "Pull Me Down" with moods powerful+energetic.
    first = next(t for t in tracks if t.track_id == "50cdd3ff-3f86-4bcf-b450-3ed022b9ba7b")
    assert "powerful" in first.moods
    assert "energetic" in first.moods
    # Genre name must not appear as a mood (separate href base).
    assert "Techno" not in first.moods


def test_discover_uses_cache_within_ttl(tmp_path, monkeypatch):
    """If a fresh catalogue.json is on disk, no network call is made."""
    from brainrotbot.music import ncs

    cached = {
        "fetched_at": 9999999999.0,  # far future -> always fresh
        "tracks": [TrackMeta(track_id="cached", title="t", artist="a", genre="g").__dict__],
    }
    cat = tmp_path / "catalogue.json"
    cat.write_text(__import__("json").dumps(cached), encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("network must not be hit when cache is fresh")
    monkeypatch.setattr(ncs, "_fetch_pages", _boom)

    out = discover_instrumental_tracks(tmp_path, ttl_days=7)
    assert [t.track_id for t in out] == ["cached"]


def test_discover_falls_back_to_stale_cache_on_scrape_failure(tmp_path, monkeypatch):
    """When scraping fails (network down), a stale catalogue beats an empty pool."""
    from brainrotbot.music import ncs

    stale = {
        "fetched_at": 0.0,  # epoch -> definitely stale
        "tracks": [TrackMeta(track_id="stale", title="t", artist="a", genre="g").__dict__],
    }
    (tmp_path / "catalogue.json").write_text(
        __import__("json").dumps(stale), encoding="utf-8"
    )
    monkeypatch.setattr(ncs, "_fetch_pages",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    out = discover_instrumental_tracks(tmp_path, ttl_days=7)
    assert [t.track_id for t in out] == ["stale"]


def test_pick_track_uses_injected_rng_for_reproducibility():
    """random.Random(seed) lets tests assert deterministic picks."""
    import random
    tracks = [TrackMeta(track_id=str(i), title=f"t{i}", artist="a", genre="g") for i in range(5)]
    pick_a = pick_track(tracks, random.Random(42))
    pick_b = pick_track(tracks, random.Random(42))
    assert pick_a.track_id == pick_b.track_id
    with pytest.raises(ValueError):
        pick_track([])
