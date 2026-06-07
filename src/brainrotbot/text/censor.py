"""Detect TikTok-sensitive words and vowel-mask them for the on-screen captions.

We no longer *replace* nasty words with euphemisms. Instead:
  * the spoken narration keeps the original word (Step 2 narrates it verbatim; a blur SFX is
    later laid over its spoken interval -- see `tts/censor.py`), and
  * the burned-in caption shows the word with its first vowel asterisked ("fuck" -> "f*ck").

This module owns the *text* half of that: loading the banned-word list, deciding whether a token
is banned (punctuation-insensitive, whole-word), masking a token's vowels, and turning a whole
cleaned story into its caption/display form plus a per-word log for the ledger. The *audio* half
(laying the blur SFX over the spoken word) lives in `tts/censor.py`; both share `is_banned`.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache

# Only the first vowel is masked (see mask_vowels). 'y' is intentionally not treated as a vowel.
_VOWELS = re.compile(r"[aeiou]", re.IGNORECASE)
# A token's "core" word, ignoring any surrounding punctuation ("fuck!" / "(fuck)" -> "fuck").
_WORD_CORE = re.compile(r"[a-z']+", re.IGNORECASE)


@lru_cache(maxsize=8)
def load_banned_words(path_str: str) -> frozenset[str]:
    """Load the lowercase banned-word set from a TOML file.

    Accepts either the new flat form (`words = ["fuck", ...]`) or the legacy `[replacements]`
    table (its keys are the words), so an old banned_words.toml keeps working. Cached by path.
    """
    with open(path_str, "rb") as f:
        data = tomllib.load(f)
    words = data.get("words") or list(data.get("replacements", {}).keys())
    return frozenset(str(w).lower() for w in words)


def is_banned(token: str, banned: frozenset[str]) -> bool:
    """True if `token`'s alphabetic core is a banned word (whole-word, case/punctuation-insensitive)."""
    m = _WORD_CORE.search(token)
    return bool(m) and m.group(0).lower() in banned


def mask_vowels(token: str) -> str:
    """Replace only the FIRST vowel with '*', leaving the rest intact ("fuck" -> "f*ck",
    "murder" -> "m*rder", "ass!" -> "*ss!"). A token with no vowels is returned unchanged."""
    return _VOWELS.sub("*", token, count=1)


def censor_for_captions(text: str, banned: frozenset[str]) -> tuple[str, list[dict]]:
    """Return (display_text, hits): the caption text with banned words vowel-masked + a per-word log.

    Whitespace is collapsed (the subtitle renderer re-splits on whitespace anyway). `hits` is a list
    of {"word", "count"} keyed by the banned word's lowercase core, recorded in the ledger for the
    Step 8 A/B analytics. An empty `banned` set (or no matches) returns the text unchanged.
    """
    counts: dict[str, int] = {}
    out: list[str] = []
    for tok in text.split():
        if is_banned(tok, banned):
            core = _WORD_CORE.search(tok).group(0).lower()
            counts[core] = counts.get(core, 0) + 1
            out.append(mask_vowels(tok))
        else:
            out.append(tok)
    hits = [{"word": w, "count": c} for w, c in counts.items()]
    return " ".join(out), hits
