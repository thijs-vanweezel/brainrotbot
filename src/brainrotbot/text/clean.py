"""Turn raw Reddit content into natural, TTS-ready prose."""

from __future__ import annotations

import html
import re

# Compiled once at import.
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")        # [text](url) -> text
_BARE_URL = re.compile(r"https?://\S+")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|\*|__|_|~~|`)")               # markdown emphasis markers
_HEADING_QUOTE = re.compile(r"^\s{0,3}([#>]+)\s?", re.MULTILINE)
_EDIT_LINE = re.compile(r"^\s*(edit|update)\s*\d*\s*[:\-].*$", re.IGNORECASE | re.MULTILINE)
_TLDR_LINE = re.compile(r"^\s*tl[;,]?\s?dr.*$", re.IGNORECASE | re.MULTILINE)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Normalize "smart" punctuation to ASCII so downstream TTS reads it cleanly.
_UNICODE_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "―": "-",
    "…": "...", " ": " ", "​": "",
}
_PUNCT_TABLE = str.maketrans(_UNICODE_PUNCT)

# HTML handling (RSS self-text arrives as rendered HTML).
_HTML_BLOCK_END = re.compile(r"(?i)</(p|div|li|h[1-6]|blockquote|tr)>")
_HTML_BR = re.compile(r"(?i)<br\s*/?>")
_HTML_TAG = re.compile(r"<[^>]+>")


def html_to_text(content: str) -> str:
    """Convert rendered HTML (e.g. RSS self-text) to plain text.

    Block-level closes and <br> become newlines; remaining tags are stripped and
    entities unescaped, then whitespace is normalized.
    """
    if not content:
        return ""
    text = _HTML_BR.sub("\n", content)
    text = _HTML_BLOCK_END.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    text = html.unescape(text)
    return _normalize_whitespace(text)


def clean_text(
    title: str,
    raw_body: str,
    *,
    prepend_title: bool = True,
    strip_edits: bool = True,
    strip_tldr: bool = True,
) -> str:
    """Normalize a Reddit post into clean prose.

    Strips markdown, links, edit/TL;DR trailers, and zero-width/control noise, then
    collapses whitespace. Optionally prepends the title as the spoken hook.
    """
    body = raw_body or ""

    if strip_edits:
        body = _EDIT_LINE.sub("", body)
    if strip_tldr:
        body = _TLDR_LINE.sub("", body)

    body = _ZERO_WIDTH.sub("", body)
    body = _MD_IMAGE.sub("", body)
    body = _MD_LINK.sub(r"\1", body)
    body = _BARE_URL.sub("", body)
    body = _HEADING_QUOTE.sub("", body)
    body = _EMPHASIS.sub("", body)

    body = _normalize_whitespace(body)

    if prepend_title and title:
        hook = _normalize_whitespace(title)
        body = f"{hook}\n\n{body}".strip() if body else hook

    return body.strip()


def _normalize_whitespace(text: str) -> str:
    # Smart quotes/dashes -> ASCII, then drop control chars (keep newlines/tabs)
    # and collapse runs of blanks.
    text = text.translate(_PUNCT_TABLE)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    # Trim trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
