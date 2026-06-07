import random
from pathlib import Path

from brainrotbot.edit.compose import pick_music_start
from brainrotbot.models import LedgerEntry, Story
from brainrotbot.music.ncs import TrackMeta
from brainrotbot.pipeline import _add_final_video, _add_music_bed


# --- _add_final_video: ledger wiring, tested against a fake editor (no ffmpeg) ----------

class _FakeEditor:
    """Stands in for VideoEditor; records the call and returns metadata without ffmpeg."""

    def __init__(self, has_music=False):
        self.calls = []
        self._has_music = has_music

    def compose(self, background_path, audio_path, out_path, music_path=None, subtitle_path=None):
        self.calls.append((str(background_path), str(audio_path), str(out_path),
                           str(music_path) if music_path else None,
                           str(subtitle_path) if subtitle_path else None))
        return {
            "path": str(out_path),
            "has_outro": True,
            "has_music": music_path is not None,
            "has_subtitles": subtitle_path is not None,
            "music_start_sec": 12.34 if music_path is not None else None,
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
    # music_path and subtitle_path are None because the test entry has neither asset.
    assert editor.calls[0] == (
        str(Path("data/video/abc123.mp4")), str(Path("data/audio/abc123.wav")),
        str(tmp_path / "final" / "abc123.mp4"), None, None,
    )


def test_add_final_video_threads_music_path_when_present(tmp_path):
    """When assets.music_path is set, compose() must receive it (-> has_music in the ledger).
    The random window start gets stamped into assets.music.start_sec for auditability."""
    entry, story = _entry()
    entry.assets["music_path"] = "data/music_cache/track.mp3"
    entry.assets["music"] = {"track_id": "x", "title": "t"}  # pre-existing from _add_music_bed
    editor = _FakeEditor()
    _add_final_video(entry, story, editor, _FakeSettings(tmp_path))
    # 4th positional in the recorded call is the music path.
    assert editor.calls[0][3] == str(Path("data/music_cache/track.mp3"))
    assert entry.assets["edit"]["has_music"] is True
    assert entry.assets["music"]["start_sec"] == 12.34


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
    """Downloads the pre-picked track (mocked) and stashes full metadata + mix settings."""
    entry, story = _entry()

    # Stub the network: produce an empty MP3 file at the expected cache path.
    from brainrotbot import pipeline as p
    def _fake_download(track, cache_dir):
        out = cache_dir / f"{track.track_id}.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-mp3")
        return out
    monkeypatch.setattr(p, "download_track", _fake_download)

    # Step 1.5: the track is pre-picked once per base story and passed in.
    _add_music_bed(entry, story, _FakeSettings(tmp_path), _track("only-one"))
    assert entry.assets["music_path"].endswith("only-one.mp3")
    m = entry.assets["music"]
    assert m["track_id"] == "only-one"
    assert m["genre"] == "Future Bass"
    assert m["moods"] == ["bright", "energetic"]
    assert m["volume_db"] == -15.0
    assert m["ducked"] is True


def test_pick_music_start_random_window_when_track_is_longer():
    """Music longer than narration -> uniform random in [intro_skip, music_dur - target]."""
    rng = random.Random(0)
    starts = {round(pick_music_start(180.0, 60.0, intro_skip_sec=5.0, rng=rng), 2)
              for _ in range(50)}
    # Must vary across calls (not always the same value) and stay inside the valid window.
    assert len(starts) > 1
    assert all(5.0 <= s <= 120.0 for s in starts)


def test_pick_music_start_returns_zero_when_track_is_too_short():
    """If music isn't even as long as the narration, no random offset is possible -- start at 0
    and let -stream_loop fill the rest."""
    assert pick_music_start(30.0, 60.0) == 0.0
    assert pick_music_start(60.0, 60.0) == 0.0  # equal length -> still 0 (no headroom)


def test_pick_music_start_clamps_intro_skip_when_window_is_tight():
    """If intro_skip exceeds the available window, the floor collapses to the window max
    so we never sample from an empty range."""
    # music=65, target=60 -> only 5s headroom; intro_skip=5 is at the boundary -> always 5.
    assert pick_music_start(65.0, 60.0, intro_skip_sec=5.0, rng=random.Random(0)) == 5.0


def test_add_music_bed_swallows_download_failure(tmp_path, monkeypatch):
    """A flaky NCS download must not block the rest of the pipeline."""
    entry, story = _entry()
    from brainrotbot import pipeline as p
    monkeypatch.setattr(p, "download_track",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _add_music_bed(entry, story, _FakeSettings(tmp_path), _track())
    assert entry.assets["music_path"] is None
    assert "music" not in entry.assets
