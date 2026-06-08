"""Detect TikTok-sensitive words and vowel-mask them for the on-screen captions.

We no longer *replace* nasty words with euphemisms. Instead:
  * the spoken narration keeps the original word (Step 2 narrates it verbatim; a blur SFX is
    later laid over its spoken interval -- see `tts/censor.py`), and
  * the burned-in caption shows the word with its first vowel asterisked ("fuck" -> "f*ck").

This module owns the *text* half of that: loading the per-language banned-word lists, deciding
whether a token is banned (punctuation-insensitive, whole-word, accent-aware), masking a token's
first vowel, and turning a cleaned story into its caption/display form plus a per-word log for the
ledger. The *audio* half (laying the blur SFX over the spoken word) lives in `tts/censor.py`; both
share `is_banned`. Translated variants (Step 1.5) censor with their own language's list -- the bot
picks `load_banned_words(...)[lang]` -- since the English words never appear in French/Spanish text.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache

# Only the FIRST vowel is masked (see mask_vowels). Accented vowels are included so French/Spanish
# words mask correctly ("pénis" -> "p*nis"); 'y' is intentionally not treated as a vowel.
_VOWELS = re.compile(r"[aeiouàáâäãèéêëìíîïòóôöõùúûü]", re.IGNORECASE)
# A token's "core" word: a run of letters of ANY script (so accents count), ignoring surrounding
# punctuation/digits ("fuck!" / "(fuck)" -> "fuck", "cocaïne." -> "cocaïne").
_WORD_CORE = re.compile(r"[^\W\d_]+", re.UNICODE)


@lru_cache(maxsize=8)
def load_banned_words(path_str: str) -> dict[str, frozenset[str]]:
    """Load the lowercase banned-word lists keyed by language from a TOML file.

    Three accepted forms (cached by path):
      * new per-language table: `[words]` with `en = [...]`, `fr = [...]`, ... -> {lang: set};
      * legacy flat list `words = [...]` -> {"en": set};
      * legacy `[replacements]` table (keys are the words) -> {"en": set}.
    A language absent from the file maps to no set, so its variant is left uncensored.
    """
    with open(path_str, "rb") as f:
        data = tomllib.load(f)
    words = data.get("words")
    if isinstance(words, dict):  # per-language table
        return {lang: frozenset(str(w).lower() for w in lst) for lang, lst in words.items()}
    if isinstance(words, list):  # legacy flat list -> English
        return {"en": frozenset(str(w).lower() for w in words)}
    if "replacements" in data:  # legacy euphemism map -> English (keys only)
        return {"en": frozenset(str(k).lower() for k in data["replacements"])}
    return {}


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
