from pathlib import Path

from brainrotbot.models import LedgerEntry, Story
from brainrotbot.music.ncs import TrackMeta
from brainrotbot.pipeline import _add_final_video, _add_music_bed


# --- _add_final_video: ledger wiring, tested against a fake editor (no ffmpeg) ----------

class _FakeEditor:
    """Stands in for VideoEditor; records the call and returns metadata without ffmpeg."""

    def __init__(self, has_music=False):
        self.calls = []
        self._has_music = has_music

    def compose(self, background_path, audio_path, out_path, music_path=None):
        self.calls.append((str(background_path), str(audio_path), str(out_path),
                           str(music_path) if music_path else None))
        return {
            "path": str(out_path),
            "has_outro": True,
            "has_music": music_path is not None,
            "outro_file": "resources/outro.mp4",
            "duration_sec": 46.0,
            "width": 1080, "height": 1920, "fps": 30,
        }


class _FakeSettings:
    def __init__(self, tmp_path):
        self.final_dir = tmp_path / "final"
        self.music_cache_dir = tmp_path / "music_cache"
        self.edit_opts = {"music_volume_db": -15.0, "music_duck": True}


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
    assert entry.assets["edit"]["has_music"] is False
    assert entry.assets["edit"]["duration_sec"] == 46.0
    # Muxes the recorded background + audio into data/final/<post_id>.mp4 (paths are
    # Path-normalized by the helper, so compare on normalized paths, not raw separators).
    # music_path is None because the test entry has no music asset.
    assert editor.calls[0] == (
        str(Path("data/video/abc123.mp4")), str(Path("data/audio/abc123.wav")),
        str(tmp_path / "final" / "abc123.mp4"), None,
    )


def test_add_final_video_threads_music_path_when_present(tmp_path):
    """When assets.music_path is set, compose() must receive it (-> has_music in the ledger)."""
    entry, story = _entry()
    entry.assets["music_path"] = "data/music_cache/track.mp3"
    editor = _FakeEditor()
    _add_final_video(entry, story, editor, _FakeSettings(tmp_path))
    # 4th positional in the recorded call is the music path.
    assert editor.calls[0][3] == str(Path("data/music_cache/track.mp3"))
    assert entry.assets["edit"]["has_music"] is True


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


# --- _add_music_bed: random pick + ledger wiring, tested without hitting NCS -----------

def _track(tid="abc"):
    return TrackMeta(
        track_id=tid, title="Demo Track", artist="DemoArtist", genre="Future Bass",
        moods=["bright", "energetic"], page_url="https://ncs.io/demo",
        instrumental_url=f"https://ncs.io/track/download/i_{tid}",
    )


def test_add_music_bed_records_track_metadata(tmp_path, monkeypatch):
    """Picks a track, downloads it (mocked), and stashes full metadata + mix settings."""
    entry, story = _entry()
    catalogue = [_track("only-one")]

    # Stub the network: produce an empty MP3 file at the expected cache path.
    from brainrotbot import pipeline as p
    def _fake_download(track, cache_dir):
        out = cache_dir / f"{track.track_id}.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-mp3")
        return out
    monkeypatch.setattr(p, "download_track", _fake_download)

    _add_music_bed(entry, story, catalogue, _FakeSettings(tmp_path))
    assert entry.assets["music_path"].endswith("only-one.mp3")
    m = entry.assets["music"]
    assert m["track_id"] == "only-one"
    assert m["genre"] == "Future Bass"
    assert m["moods"] == ["bright", "energetic"]
    assert m["volume_db"] == -15.0
    assert m["ducked"] is True


def test_add_music_bed_swallows_download_failure(tmp_path, monkeypatch):
    """A flaky NCS download must not block the rest of the pipeline."""
    entry, story = _entry()
    from brainrotbot import pipeline as p
    monkeypatch.setattr(p, "download_track",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _add_music_bed(entry, story, [_track()], _FakeSettings(tmp_path))
    assert entry.assets["music_path"] is None
    assert "music" not in entry.assets
