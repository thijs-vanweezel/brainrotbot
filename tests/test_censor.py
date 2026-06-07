from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from brainrotbot.text.censor import (
    censor_for_captions,
    is_banned,
    load_banned_words,
    mask_vowels,
)
from brainrotbot.tts.censor import BlurCensor


def _write_words(tmp_path: Path) -> Path:
    p = tmp_path / "banned.toml"
    p.write_text('words = ["fuck", "kill", "gun"]\n', encoding="utf-8")
    return p


# --- text side: detection + masking --------------------------------------------------

def test_load_supports_words_list_and_legacy_replacements(tmp_path):
    new = _write_words(tmp_path)
    assert load_banned_words(str(new)) == frozenset({"fuck", "kill", "gun"})
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[replacements]\nkill = "unalive"\ngun = "pew-pew"\n', encoding="utf-8")
    assert load_banned_words(str(legacy)) == frozenset({"kill", "gun"})


def test_mask_vowels_masks_only_first_vowel():
    assert mask_vowels("fuck") == "f*ck"
    assert mask_vowels("FUCK!") == "F*CK!"
    assert mask_vowels("shit") == "sh*t"
    # Multi-vowel words: only the FIRST vowel is asterisked, the rest stay.
    assert mask_vowels("murder") == "m*rder"
    assert mask_vowels("cocaine") == "c*caine"
    assert mask_vowels("ass!") == "*ss!"
    assert mask_vowels("rhythm") == "rhythm"  # no a/e/i/o/u -> unchanged


def test_is_banned_whole_word_and_punctuation_insensitive():
    banned = frozenset({"fuck", "kill"})
    assert is_banned("fuck", banned)
    assert is_banned("Fuck!", banned)        # case + trailing punctuation ignored
    assert is_banned("(fuck)", banned)
    assert not is_banned("skill", banned)    # "kill" is a substring -> must NOT match
    assert not is_banned("ducks", banned)


def test_censor_for_captions_masks_and_logs():
    banned = frozenset({"fuck", "gun"})
    display, hits = censor_for_captions("I grabbed the gun and said fuck. Fuck!", banned)
    assert "gun" not in display and "fuck" not in display.lower().replace("f*ck", "")
    assert "g*n" in display and "f*ck" in display and "F*ck" in display
    by = {h["word"]: h["count"] for h in hits}
    assert by == {"gun": 1, "fuck": 2}


def test_censor_for_captions_noop_when_clean():
    display, hits = censor_for_captions("a perfectly clean sentence", frozenset({"fuck"}))
    assert display == "a perfectly clean sentence"
    assert hits == []


# --- audio side: blur overlay --------------------------------------------------------

def _make_sfx(tmp_path: Path) -> Path:
    """A tiny 1 kHz tone WAV used as the blur SFX (BlurCensor decodes it via ffmpeg)."""
    sr = 24000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sf.write(tmp_path / "sfx.wav", (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), sr)
    return tmp_path / "sfx.wav"


def test_blur_masks_the_word_interval(tmp_path):
    sr = 24000
    # 1 s of loud full-scale narration; blur the middle 0.4-0.6 s window.
    audio = np.ones(sr, dtype=np.float32) * 0.9
    sf.write(tmp_path / "n.wav", audio, sr)
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), voice_duck_db=-40, pad_sec=0.0,
                      sample_rate=sr)

    n = blur.censor(tmp_path / "n.wav", [(0.4, 0.6)])
    assert n == 1
    out, _ = sf.read(tmp_path / "n.wav", dtype="float32")
    assert len(out) == sr  # length unchanged

    s, e = int(0.4 * sr), int(0.6 * sr)
    inner = out[s + 300:e - 300]  # away from the edge fades
    # The narration's loud DC (0.9) is ducted away, so the window's mean collapses toward 0 (the
    # added tone is zero-mean); yet the window is loud (the peak-normalized blur dominates).
    assert abs(float(np.mean(inner))) < 0.05
    assert float(np.max(np.abs(inner))) > 0.5
    # before/after the window: untouched (abs tol covers the WAV's 16-bit quantization).
    assert out[100] == pytest.approx(0.9, abs=1e-3)
    assert out[-100] == pytest.approx(0.9, abs=1e-3)


def test_blur_noop_without_intervals(tmp_path):
    sr = 24000
    sf.write(tmp_path / "n.wav", np.ones(sr, dtype=np.float32) * 0.5, sr)
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), sample_rate=sr)
    assert blur.censor(tmp_path / "n.wav", []) == 0
