import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from brainrotbot.models import LedgerEntry, Story
from brainrotbot.pipeline import _add_audio
from brainrotbot.tts.intro_sfx import IntroSfx
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


# --- IntroSfx: overlay the ahem after the title, length preserved -----------------

def _make_tone(tmp_path: Path) -> Path:
    """A tiny 1 kHz tone WAV used as the ahem SFX (IntroSfx decodes it via ffmpeg)."""
    sr = 24000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sf.write(tmp_path / "ahem.wav", (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), sr)
    return tmp_path / "ahem.wav"


def test_intro_sfx_overlays_at_timestamp_keeping_length(tmp_path):
    sr = 24000
    # 1 s of quiet narration; overlay the ahem at the 0.4 s "title end".
    sf.write(tmp_path / "n.wav", np.full(sr, 0.1, dtype=np.float32), sr)
    fx = IntroSfx(sfx_file=str(_make_tone(tmp_path)), sample_rate=sr)

    assert fx.overlay(tmp_path / "n.wav", 0.4) is True
    out, _ = sf.read(tmp_path / "n.wav", dtype="float32")
    assert len(out) == sr  # length unchanged (pure overlay, truncated to fit)

    # Before the start: untouched narration. After: the loud (peak ~0.95) tone is layered on top.
    assert out[100] == pytest.approx(0.1, abs=1e-3)
    s = int(0.4 * sr)
    window = out[s + 300 : s + int(0.2 * sr) - 300]  # inside the 0.2 s tone, past the edge fade
    assert float(np.max(np.abs(window))) > 0.8  # clearly audible over the narration


def test_intro_sfx_noop_when_start_past_end(tmp_path):
    sr = 24000
    sf.write(tmp_path / "n.wav", np.full(sr, 0.1, dtype=np.float32), sr)
    fx = IntroSfx(sfx_file=str(_make_tone(tmp_path)), sample_rate=sr)
    assert fx.overlay(tmp_path / "n.wav", 2.0) is False  # 2 s into a 1 s clip -> nothing to do
    out, _ = sf.read(tmp_path / "n.wav", dtype="float32")
    assert np.allclose(out, 0.1, atol=1e-3)  # WAV left unchanged
