"""Replace TikTok-sensitive words with community euphemisms.

Whole-word, case-insensitive matching; replacement casing follows the original token
(UPPER, Title, or lower). Returns the filtered text plus a log of what was changed so
the ledger can record it.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

from ..models import Replacement


@lru_cache(maxsize=8)
def _load_map(path_str: str) -> tuple[tuple[str, str], ...]:
    with open(path_str, "rb") as f:
        data = tomllib.load(f)
    mapping = data.get("replacements", {})
    # Longest source words first so multi-token-ish entries win before substrings.
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    return tuple((str(k), str(v)) for k, v in items)


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def filter_banned_words(text: str, banned_words_path: Path) -> tuple[str, list[Replacement]]:
    replacements: list[Replacement] = []
    result = text
    for src, dst in _load_map(str(banned_words_path)):
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        count = 0

        def _sub(m: re.Match, _dst=dst) -> str:
            nonlocal count
            count += 1
            return _match_case(m.group(0), _dst)

        result, n = pattern.subn(_sub, result)
        if n:
            replacements.append(Replacement(from_word=src, to_word=dst, count=count))
    return result, replacements
