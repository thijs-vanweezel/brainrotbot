import random

import pytest

from brainrotbot.models import LedgerEntry, Story
from brainrotbot.pipeline import _add_background_video
from brainrotbot.video.background import _is_direct_media_url, ensure_cached, pick_source


# --- direct-media sources: the yt-dlp-free path CI relies on ----------------------

@pytest.mark.parametrize("url", [
    "https://github.com/o/r/releases/download/sources-v1/subway-surfers-1.mp4",
    "https://example.com/a.WEBM",                     # extension match is case-insensitive
    "https://example.com/a.mkv?token=abc&x=1",        # query string is not part of the path
])
def test_is_direct_media_url_true(url):
    assert _is_direct_media_url(url)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=i0M4ARe9v0Y",    # the yt-dlp path
    "https://youtu.be/i0M4ARe9v0Y",
    "https://example.com/video",                      # no extension -> needs an extractor
])
def test_is_direct_media_url_false(url):
    assert not _is_direct_media_url(url)


def test_ensure_cached_downloads_direct_url_without_ytdlp(tmp_path, monkeypatch):
    """A direct media URL is streamed straight to the cache -- yt-dlp is never resolved."""
    monkeypatch.setattr("brainrotbot.video.background._download",
                        lambda url, dest: dest.write_bytes(b"fake mp4"))
    monkeypatch.setattr("brainrotbot.video.background._require",
                        lambda b: pytest.fail(f"{b} must not be needed for a direct URL"))
    got = ensure_cached("https://host/x/subway.mp4", tmp_path, max_height=1080)
    assert got.read_bytes() == b"fake mp4" and got.suffix == ".mp4"
    # A second call reuses the cached file rather than re-downloading.
    monkeypatch.setattr("brainrotbot.video.background._download",
                        lambda url, dest: pytest.fail("should have hit the cache"))
    assert ensure_cached("https://host/x/subway.mp4", tmp_path, max_height=1080) == got


# --- pick_source: uniform random pick over the source pool ------------------------

def test_pick_source_picks_from_pool_and_is_seed_deterministic():
    sources = ["a", "b", "c"]
    # Every pick is a pool member.
    assert all(pick_source(sources, random.Random(i)) in sources for i in range(20))
    # A given seed reproduces the same pick (injectable rng -> testable).
    assert pick_source(sources, random.Random(42)) == pick_source(sources, random.Random(42))


def test_pick_source_empty_pool_raises():
    with pytest.raises(ValueError):
        pick_source([])


# --- _add_background_video: ledger wiring, tested against a fake maker -------------

class _FakeMaker:
    """Stands in for BackgroundVideoMaker; records the call and returns metadata without
    any network/ffmpeg or writing a real clip."""

    def __init__(self):
        self.calls = []

    def make(self, source_url, duration_sec, out_path):
        self.calls.append((source_url, duration_sec, str(out_path)))
        return {
            "path": str(out_path),
            "source_url": source_url,
            "source_id": "vid123",
            "start_sec": 4.0,
            "duration_sec": duration_sec,
            "looped": False,
            "width": 1080, "height": 1920, "fps": 30,
        }


class _FakeSettings:
    def __init__(self, tmp_path, sources=("https://yt/a", "https://yt/b")):
        self.video_dir = tmp_path / "video"
        self.video_opts = {"sources": list(sources)}


def _entry(with_audio=True):
    story = Story(
        post_id="abc123", subreddit="tifu", title="T", raw_body="b",
        url="u", author="x", created_utc=0.0,
    )
    entry = LedgerEntry.from_story(story, "the cleaned narration text", [])
    if with_audio:
        entry.assets["audio"] = {"voice": "af_heart", "lang_code": "a",
                                 "duration_sec": 42.0, "sample_rate": 24000}
    return entry, story


def test_add_background_video_records_assets_and_status(tmp_path):
    entry, story = _entry()
    maker = _FakeMaker()
    _add_background_video(entry, story, maker, _FakeSettings(tmp_path))

    assert entry.status == "video_done"
    assert entry.assets["background_video"].endswith("abc123.mp4")
    assert entry.assets["background"]["source_id"] == "vid123"
    assert entry.assets["background"]["start_sec"] == 4.0
    # Trims to the real narrated duration; source is randomly drawn from the pool.
    url, duration, out = maker.calls[0]
    assert url in ("https://yt/a", "https://yt/b")
    assert (duration, out) == (42.0, str(tmp_path / "video" / "abc123.mp4"))


def test_add_background_video_falls_back_to_estimate_without_audio(tmp_path):
    entry, story = _entry(with_audio=False)
    maker = _FakeMaker()
    _add_background_video(entry, story, maker, _FakeSettings(tmp_path))
    # No audio -> uses the word-count estimate from the cleaned text.
    assert maker.calls[0][1] == entry.text["est_speech_seconds"]


def test_add_background_video_picks_source_from_pool(tmp_path):
    settings = _FakeSettings(tmp_path)
    for _ in range(5):
        entry, story = _entry()
        maker = _FakeMaker()
        _add_background_video(entry, story, maker, settings)
        assert maker.calls[0][0] in ("https://yt/a", "https://yt/b")


def test_add_background_video_no_sources_skips(tmp_path):
    entry, story = _entry()
    _add_background_video(entry, story, _FakeMaker(), _FakeSettings(tmp_path, sources=()))
    assert entry.status == "cleaned"  # unchanged (not advanced to video_done), background left null
    assert entry.assets["background_video"] is None


def test_add_background_video_swallows_failure(tmp_path):
    """A render/download failure must not abort the run: background_video stays null."""
    class _Boom:
        def make(self, *a, **k):
            raise RuntimeError("ffmpeg exploded")

    entry, story = _entry()
    _add_background_video(entry, story, _Boom(), _FakeSettings(tmp_path))
    assert entry.assets["background_video"] is None
