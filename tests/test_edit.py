from pathlib import Path

from brainrotbot.models import LedgerEntry, Story
from brainrotbot.pipeline import _add_final_video


# --- _add_final_video: ledger wiring, tested against a fake editor (no ffmpeg) ----------

class _FakeEditor:
    """Stands in for VideoEditor; records the call and returns metadata without ffmpeg."""

    def __init__(self):
        self.calls = []

    def compose(self, background_path, audio_path, out_path):
        self.calls.append((str(background_path), str(audio_path), str(out_path)))
        return {
            "path": str(out_path),
            "has_outro": True,
            "outro_file": "resources/outro.mp4",
            "duration_sec": 46.0,
            "width": 1080, "height": 1920, "fps": 30,
        }


class _FakeSettings:
    def __init__(self, tmp_path):
        self.final_dir = tmp_path / "final"


def _entry(*, with_audio=True, with_background=True):
    story = Story(
        post_id="abc123", subreddit="tifu", title="T", raw_body="b",
        url="u", author="x", created_utc=0.0,
    )
    entry = LedgerEntry.from_story(story, "the cleaned narration text", [])
    if with_audio:
        entry.assets["audio_path"] = "data/audio/abc123.wav"
    if with_background:
        entry.assets["background_video"] = "data/video/abc123.mp4"
    return entry, story


def test_add_final_video_records_assets_and_status(tmp_path):
    entry, story = _entry()
    editor = _FakeEditor()
    _add_final_video(entry, story, editor, _FakeSettings(tmp_path))

    assert entry.status == "edit_done"
    assert entry.assets["final_video"].endswith("abc123.mp4")
    assert entry.assets["edit"]["has_outro"] is True
    assert entry.assets["edit"]["duration_sec"] == 46.0
    # Muxes the recorded background + audio into data/final/<post_id>.mp4 (paths are
    # Path-normalized by the helper, so compare on normalized paths, not raw separators).
    assert editor.calls[0] == (
        str(Path("data/video/abc123.mp4")), str(Path("data/audio/abc123.wav")),
        str(tmp_path / "final" / "abc123.mp4"),
    )


def test_add_final_video_skips_without_background(tmp_path):
    entry, story = _entry(with_background=False)
    editor = _FakeEditor()
    _add_final_video(entry, story, editor, _FakeSettings(tmp_path))
    assert editor.calls == []                       # not invoked
    assert entry.status == "cleaned"                # unchanged
    assert entry.assets["final_video"] is None


def test_add_final_video_skips_without_audio(tmp_path):
    entry, story = _entry(with_audio=False)
    editor = _FakeEditor()
    _add_final_video(entry, story, editor, _FakeSettings(tmp_path))
    assert editor.calls == []
    assert entry.assets["final_video"] is None


def test_add_final_video_swallows_failure(tmp_path):
    """An ffmpeg failure must not abort the run: final_video stays null."""
    class _Boom:
        def compose(self, *a, **k):
            raise RuntimeError("ffmpeg exploded")

    entry, story = _entry()
    _add_final_video(entry, story, _Boom(), _FakeSettings(tmp_path))
    assert entry.assets["final_video"] is None
