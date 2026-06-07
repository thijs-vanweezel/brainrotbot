import random

import pytest

from brainrotbot.models import LedgerEntry, Story
from brainrotbot.pipeline import _add_audio
from brainrotbot.tts.synthesize import pick_voice


# --- pick_voice: uniform random pick over a language's pool -----------------------

def test_pick_voice_picks_from_pool_and_is_seed_deterministic():
    voices = ["a", "b", "c"]
    # Every pick is a pool member.
    assert all(pick_voice(voices, random.Random(i)) in voices for i in range(20))
    # A given seed reproduces the same pick (injectable rng -> testable).
    assert pick_voice(voices, random.Random(42)) == pick_voice(voices, random.Random(42))


def test_pick_voice_empty_pool_raises():
    with pytest.raises(ValueError):
        pick_voice([])


# --- _add_audio: ledger wiring, tested against a fake synthesizer -----------------

class _FakeSynth:
    """Stands in for KokoroSynthesizer; records the call and returns metadata without
    loading the model or writing real audio."""

    def __init__(self):
        self.calls = []

    def synthesize(self, text, out_path, *, voice, lang_code, speed):
        self.calls.append((text, str(out_path), voice, lang_code, speed))
        return {
            "audio_path": str(out_path),
            "voice": voice,
            "lang_code": lang_code,
            "duration_sec": 12.34,
            "sample_rate": 24000,
        }


class _FakeSettings:
    def __init__(self, tmp_path):
        self.audio_dir = tmp_path / "audio"
        self.tts_opts = {
            "default_lang": "a",
            "speed": 1.0,
            "voices": {"a": ["af_heart", "am_michael"]},
        }


def _entry():
    story = Story(
        post_id="abc123", subreddit="tifu", title="T", raw_body="b",
        url="u", author="x", created_utc=0.0,
    )
    return LedgerEntry.from_story(story, "the cleaned narration text", []), story


def test_add_audio_records_assets_and_status(tmp_path):
    entry, story = _entry()
    synth = _FakeSynth()
    _add_audio(entry, story, synth, _FakeSettings(tmp_path))

    assert entry.status == "tts_done"
    assert entry.assets["audio_path"].endswith("abc123.wav")
    # Voice is randomly drawn from the configured pool.
    assert entry.assets["audio"]["voice"] in ("af_heart", "am_michael")
    assert {k: entry.assets["audio"][k] for k in ("lang_code", "duration_sec", "sample_rate")} == {
        "lang_code": "a", "duration_sec": 12.34, "sample_rate": 24000,
    }
    # The cleaned body (not the title/raw) is what gets narrated.
    assert synth.calls[0][0] == "the cleaned narration text"


def test_add_audio_picks_voice_from_pool(tmp_path):
    settings = _FakeSettings(tmp_path)
    for _ in range(5):
        entry, story = _entry()
        _add_audio(entry, story, _FakeSynth(), settings)
        assert entry.assets["audio"]["voice"] in ("af_heart", "am_michael")


def test_add_audio_swallows_failure(tmp_path):
    """A TTS failure must not abort the run: audio_path stays null, status unchanged."""
    class _Boom:
        def synthesize(self, *a, **k):
            raise RuntimeError("model exploded")

    entry, story = _entry()
    _add_audio(entry, story, _Boom(), _FakeSettings(tmp_path))
    assert entry.status == "cleaned"
    assert entry.assets["audio_path"] is None
