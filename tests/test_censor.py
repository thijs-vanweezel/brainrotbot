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

def test_load_per_language_table(tmp_path):
    p = tmp_path / "multi.toml"
    p.write_text('[words]\nen = ["fuck"]\nfr = ["putain", "merde"]\n', encoding="utf-8")
    by_lang = load_banned_words(str(p))
    assert by_lang["en"] == frozenset({"fuck"})
    assert by_lang["fr"] == frozenset({"putain", "merde"})
    assert by_lang.get("es", frozenset()) == frozenset()  # absent language -> uncensored


def test_load_supports_legacy_flat_list_and_replacements(tmp_path):
    flat = _write_words(tmp_path)  # words = [...] -> English
    assert load_banned_words(str(flat)) == {"en": frozenset({"fuck", "kill", "gun"})}
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[replacements]\nkill = "unalive"\ngun = "pew-pew"\n', encoding="utf-8")
    assert load_banned_words(str(legacy)) == {"en": frozenset({"kill", "gun"})}


def test_detection_and_masking_with_accents():
    # French/Spanish words with accents must be detected (Unicode word-core) and have their FIRST
    # vowel masked, including when that vowel is accented.
    banned = frozenset({"putain", "merde", "cocaïne", "pénis"})
    assert is_banned("Putain!", banned)
    assert is_banned("cocaïne.", banned)      # accent kept in the core, not split on 'ï'
    assert mask_vowels("putain") == "p*tain"
    assert mask_vowels("pénis") == "p*nis"    # first vowel is the accented 'é'
    display, hits = censor_for_captions("c'est de la merde", banned)
    assert "m*rde" in display
    assert {h["word"]: h["count"] for h in hits} == {"merde": 1}


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
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), voice_duck_db=-40, sfx_inset_sec=0.0,
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


def test_blur_inset_runs_shorter_than_the_word(tmp_path):
    sr = 24000
    audio = np.ones(sr, dtype=np.float32) * 0.9
    sf.write(tmp_path / "n.wav", audio, sr)
    # Inset 0.05 s each edge: a 0.4-0.6 s word -> blur/duck only over ~0.45-0.55 s.
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), voice_duck_db=-40, sfx_inset_sec=0.05,
                      sample_rate=sr)
    assert blur.censor(tmp_path / "n.wav", [(0.4, 0.6)]) == 1
    out, _ = sf.read(tmp_path / "n.wav", dtype="float32")
    # The word edges (just inside 0.4 / 0.6) are NOT touched -- the beep is shorter than the word.
    assert out[int(0.41 * sr)] == pytest.approx(0.9, abs=1e-3)
    assert out[int(0.59 * sr)] == pytest.approx(0.9, abs=1e-3)
    # ...but the inset centre IS blurred (loud, zero-mean) -- the word is still masked there.
    centre = out[int(0.48 * sr):int(0.52 * sr)]
    assert abs(float(np.mean(centre))) < 0.05 and float(np.max(np.abs(centre))) > 0.3


def test_blur_too_short_word_falls_back_to_exact_interval(tmp_path):
    sr = 24000
    sf.write(tmp_path / "n.wav", np.ones(sr, dtype=np.float32) * 0.9, sr)
    # A 0.03 s word with a 0.05 s inset would collapse -> fall back to the exact interval so the
    # banned word is never left audible.
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), voice_duck_db=-40, sfx_inset_sec=0.05,
                      sample_rate=sr)
    assert blur.censor(tmp_path / "n.wav", [(0.40, 0.43)]) == 1
    out, _ = sf.read(tmp_path / "n.wav", dtype="float32")
    centre = out[int(0.41 * sr):int(0.42 * sr)]
    assert abs(float(np.mean(centre))) < 0.1  # the short word is ducked/masked, not skipped


def test_blur_noop_without_intervals(tmp_path):
    sr = 24000
    sf.write(tmp_path / "n.wav", np.ones(sr, dtype=np.float32) * 0.5, sr)
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), sample_rate=sr)
    assert blur.censor(tmp_path / "n.wav", []) == 0


def test_blur_capped_at_max_sfx_sec(tmp_path):
    """The audible beep must be ≤ max_sfx_sec; the full duck span still masks the word."""
    sr = 24000
    # 2 s of loud narration; blur a long interval (1.0-1.8 s = 0.8 s span) with a 0.30 s cap.
    audio = np.ones(sr * 2, dtype=np.float32) * 0.9
    sf.write(tmp_path / "n.wav", audio, sr)
    blur = BlurCensor(sfx_file=str(_make_sfx(tmp_path)), voice_duck_db=-40,
                      sfx_inset_sec=0.0, max_sfx_sec=0.30, sample_rate=sr)
    assert blur.censor(tmp_path / "n.wav", [(1.0, 1.8)]) == 1
    out, _ = sf.read(tmp_path / "n.wav", dtype="float32")
    cap = int(0.30 * sr)
    # Leading cap: the beep region is loud and zero-mean (tone).
    beep_region = out[int(1.0 * sr) + 50: int(1.0 * sr) + cap - 50]
    assert float(np.max(np.abs(beep_region))) > 0.3
    # Post-cap within the duck span: narration is ducked near-silence, no beep energy.
    ducked_tail = out[int(1.0 * sr) + cap + 100: int(1.8 * sr) - 100]
    assert float(np.max(np.abs(ducked_tail))) < 0.1
    # After the span: narration is untouched.
    assert out[int(1.9 * sr)] == pytest.approx(0.9, abs=1e-3)


def test_new_en_tokens_are_detected():
    """New terms added to the en list must be detected by is_banned (smoke-check a sample)."""
    banned = frozenset({
        "stabbed", "overdose", "terrorist", "pedophile", "incest", "meth",
        "strangle", "corpse", "wanker", "bullshit", "nude", "orgasm",
    })
    for word in banned:
        assert is_banned(word, banned), f"expected {word!r} to be detected"
    # Punctuation-wrapped forms also match:
    assert is_banned("stabbed!", banned)
    assert is_banned("(overdose)", banned)
