"""Tests for the Step 1.5 translator.

The model-backed tests are heavy (they convert + load NLLB), so they're marked and skipped unless
ctranslate2 is installed AND BRBOT_TEST_TRANSLATE=1 is set -- keeping the default suite fast/offline.
The lang-map + helper tests always run (pure logic, no model).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from brainrotbot.text import translate as T


def test_language_map_codes():
    # Every language maps to a single-char Kokoro lang_code; only "en" has no NLLB target.
    assert T.LANGUAGES["en"]["nllb"] == ""
    assert T.kokoro_lang_code("fr") == "f"
    assert T.kokoro_lang_code("es") == "e"
    assert T.kokoro_lang_code("zh") == "z"
    assert T.kokoro_lang_code("unknown") == "a"  # falls back to American English


def test_split_sentences():
    parts = T._split_sentences("Hello there. How are you? I am fine!\nNew para.")
    assert parts == ["Hello there.", "How are you?", "I am fine!", "New para."]


def test_english_is_passthrough_without_model():
    # "en" must never load the model (translator unused), so a bogus cache dir is fine.
    tr = T.Translator(cache_dir=Path("/nonexistent"))
    assert tr.translate("Keep me as is.", "en") == "Keep me as is."
    assert tr.translate("", "fr") == ""  # empty stays empty, still no model load


def test_localize_preserves_paragraph_break():
    """_localize must keep the '\\n\\n' title/body separator so intro_words > 0 for fr/es."""
    from dataclasses import dataclass, field
    from brainrotbot.pipeline import _localize

    @dataclass
    class _FakeTr:
        model_label = "fake"
        def translate(self, text: str, lang: str) -> str:
            return f"[{lang}]{text}"

    @dataclass
    class _Entry:
        source: dict = field(default_factory=dict)
        text: dict = field(default_factory=lambda: {
            "title": "My Title",
            "cleaned_body": "My Title\n\nBody text here.",
        })

    entry = _Entry()
    _localize(entry, "fr", _FakeTr())
    # The translated cleaned_body must retain a \n\n so the pipeline can find intro_words.
    assert "\n\n" in entry.text["cleaned_body"]
    # The first paragraph is the translated title; the second is the translated body.
    title_part, _, body_part = entry.text["cleaned_body"].partition("\n\n")
    assert title_part == "[fr]My Title"
    assert body_part == "[fr]Body text here."
    # title field is also updated.
    assert entry.text["title"] == "[fr]My Title"


def test_localize_title_only_no_break():
    """Title-only stories (no \\n\\n) degrade cleanly: cleaned_body = translated title."""
    from dataclasses import dataclass, field
    from brainrotbot.pipeline import _localize

    @dataclass
    class _FakeTr:
        model_label = "fake"
        def translate(self, text: str, lang: str) -> str:
            return f"[{lang}]{text}"

    @dataclass
    class _Entry:
        source: dict = field(default_factory=dict)
        text: dict = field(default_factory=lambda: {
            "title": "Just a title",
            "cleaned_body": "Just a title",
        })

    entry = _Entry()
    _localize(entry, "es", _FakeTr())
    assert entry.text["cleaned_body"] == "[es]Just a title"
    assert "\n\n" not in entry.text["cleaned_body"]


needs_model = pytest.mark.skipif(
    not (os.environ.get("BRBOT_TEST_TRANSLATE") == "1"),
    reason="set BRBOT_TEST_TRANSLATE=1 (and install ctranslate2) to run model-backed translation",
)


@needs_model
@pytest.mark.parametrize("lang", ["fr", "es", "zh"])
def test_translation_changes_text(lang):
    tr = T.Translator(cache_dir=Path("data/translation_cache"))
    src = "Hello, my name is John. I love telling stories."
    out = tr.translate(src, lang)
    assert out and out != src           # produced something, and it's not the English source
    assert "<unk>" not in out           # unmappable tokens are stripped
