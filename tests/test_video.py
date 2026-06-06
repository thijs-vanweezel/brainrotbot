import pytest

from brainrotbot.models import LedgerEntry, Story
from brainrotbot.pipeline import _add_background_video
from brainrotbot.video.background import pick_source


# --- pick_source: deterministic round-robin over the source pool -----------------

def test_pick_source_round_robins_and_wraps():
    sources = ["a", "b", "c"]
    got = [pick_source(sources, i) for i in range(5)]
    assert got == ["a", "b", "c", "a", "b"]


def test_pick_source_empty_pool_raises():
    with pytest.raises(ValueError):
        pick_source([], 0)


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
    _add_background_video(entry, story, maker, _FakeSettings(tmp_path), index=0)

    assert entry.status == "video_done"
    assert entry.assets["background_video"].endswith("abc123.mp4")
    assert entry.assets["background"]["source_id"] == "vid123"
    assert entry.assets["background"]["start_sec"] == 4.0
    # Trims to the real narrated duration, and rotates source 0.
    assert maker.calls[0] == ("https://yt/a", 42.0, str(tmp_path / "video" / "abc123.mp4"))


def test_add_background_video_falls_back_to_estimate_without_audio(tmp_path):
    entry, story = _entry(with_audio=False)
    maker = _FakeMaker()
    _add_background_video(entry, story, maker, _FakeSettings(tmp_path), index=0)
    # No audio -> uses the word-count estimate from the cleaned text.
    assert maker.calls[0][1] == entry.text["est_speech_seconds"]


def test_add_background_video_rotates_source_by_index(tmp_path):
    settings = _FakeSettings(tmp_path)
    used = []
    for i in range(3):
        entry, story = _entry()
        maker = _FakeMaker()
        _add_background_video(entry, story, maker, settings, index=i)
        used.append(maker.calls[0][0])
    assert used == ["https://yt/a", "https://yt/b", "https://yt/a"]


def test_add_background_video_no_sources_skips(tmp_path):
    entry, story = _entry()
    _add_background_video(entry, story, _FakeMaker(), _FakeSettings(tmp_path, sources=()), index=0)
    assert entry.status == "cleaned"  # unchanged (not advanced to video_done), background left null
    assert entry.assets["background_video"] is None


def test_add_background_video_swallows_failure(tmp_path):
    """A render/download failure must not abort the run: background_video stays null."""
    class _Boom:
        def make(self, *a, **k):
            raise RuntimeError("ffmpeg exploded")

    entry, story = _entry()
    _add_background_video(entry, story, _Boom(), _FakeSettings(tmp_path), index=0)
    assert entry.assets["background_video"] is None
