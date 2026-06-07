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
# Reddit's RSS <content> always tails the body with two structural lines:
#   "submitted by /u/<author>" and "[link] [comments]". Both are plain text after
#   html_to_text(), so nothing else strips them. Match each line independently so
#   we don't depend on order/spacing between them. False-positive risk in real prose
#   is essentially nil.
_REDDIT_RSS_FOOTER = re.compile(
    r"^\s*(submitted by /?u/\S+|\[link\]\s+\[comments?\])\s*$",
    re.IGNORECASE | re.MULTILINE,
)
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

# Reddit-isms expanded into full prose before TTS so Kokoro reads "AITA" as words, not letters.
# Case-sensitive (acronyms are reliably uppercase on Reddit); word-boundary anchored so legit
# words containing the letters survive (e.g. "drop" -> not expanded as "drOP"). Skipped on
# purpose: AH (collides with "ah" interjection), ATM (often means cash machine), INFO (common
# noun), F/M alone (ambiguous outside age/gender form).
_REDDIT_ABBREVIATIONS = {
    # AmItheAsshole / TIFU verdicts and tags. AITAH/WIBTAH listed because the regex below sorts
    # by length so the longer match wins.
    "AITAH": "Am I the asshole here",
    "AITA": "Am I the asshole",
    "WIBTAH": "Would I be the asshole here",
    "WIBTA": "Would I be the asshole",
    "YTA": "You're the asshole",
    "NTA": "Not the asshole",
    "ESH": "Everyone sucks here",
    "NAH": "No assholes here",
    "TIFU": "Today I fucked up",
    "OP": "the original poster",
    # Relationship / family shorthand common in r/relationship_advice, r/AITA, etc.
    "SO": "significant other",
    "BF": "boyfriend",
    "GF": "girlfriend",
    "DH": "husband",
    "DW": "wife",
    "DS": "son",
    "DD": "daughter",
    "MIL": "mother-in-law",
    "FIL": "father-in-law",
    "SIL": "sister-in-law",
    "BIL": "brother-in-law",
    "STBXH": "soon-to-be ex-husband",
    "STBXW": "soon-to-be ex-wife",
    "STBX": "soon-to-be ex",
    "SAHM": "stay-at-home mom",
    "SAHD": "stay-at-home dad",
    "LO": "little one",
    "AP": "affair partner",
    "LDAP": "long-distance affair partner",
    # General internet shorthand
    "IMHO": "in my humble opinion",
    "IMO": "in my opinion",
    "TBH": "to be honest",
    "IRL": "in real life",
    "IDK": "I don't know",
    "AFAIK": "as far as I know",
    "IIRC": "if I recall correctly",
    "FWIW": "for what it's worth",
    "ELI5": "explain like I'm five",
    "FYI": "for your information",
    "NSFW": "not safe for work",
    "ASAP": "as soon as possible",
    "DM": "direct message",
    "LPT": "life pro tip",
    "ETA": "edited to add",
    "IDGAF": "I don't give a fuck",
    "IDC": "I don't care",
}
# Single alternation regex, length-descending so AITAH wins over AITA.
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(k) for k in sorted(_REDDIT_ABBREVIATIONS, key=len, reverse=True)
    ) + r")\b"
)
# "(39F)" / "(F39)" / "(28 m)" age+gender markers -> "39 year old female". Restricted to inside
# parens because that's the only form where false-positives (e.g. "F35 hours") are impossible.
_AGE_GENDER_AGE_FIRST = re.compile(r"\((\d{1,2})\s*([MFmf])\)")
_AGE_GENDER_SEX_FIRST = re.compile(r"\(([MFmf])\s*(\d{1,2})\)")


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


def expand_reddit_abbreviations(text: str) -> str:
    """Expand AITA/BF/MIL/(39F)/... into prose so Kokoro narrates words, not letters."""
    text = _ABBREV_RE.sub(lambda m: _REDDIT_ABBREVIATIONS[m.group(1)], text)
    text = _AGE_GENDER_AGE_FIRST.sub(
        lambda m: f"{m.group(1)} year old {'male' if m.group(2).upper() == 'M' else 'female'}",
        text,
    )
    text = _AGE_GENDER_SEX_FIRST.sub(
        lambda m: f"{m.group(2)} year old {'male' if m.group(1).upper() == 'M' else 'female'}",
        text,
    )
    return text


def clean_text(
    title: str,
    raw_body: str,
    *,
    prepend_title: bool = True,
    strip_edits: bool = True,
    strip_tldr: bool = True,
) -> str:
    """Normalize a Reddit post into clean prose.

    Strips Reddit's RSS footer ("submitted by /u/..." + "[link] [comments]"), markdown,
    links, edit/TL;DR trailers, and zero-width/control noise; expands Reddit abbreviations
    (AITA, BF, MIL, "(39F)", ...) so the TTS reads them as words; collapses whitespace.
    Optionally prepends the title as the spoken hook.
    """
    body = raw_body or ""

    # Footer first: it can sit adjacent to a final "Edit:" line, and stripping it before
    # _EDIT_LINE lets _normalize_whitespace later collapse the gap cleanly.
    body = _REDDIT_RSS_FOOTER.sub("", body)
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

    # Expand abbreviations last so it runs on the assembled title+body in one shot.
    return expand_reddit_abbreviations(body.strip())


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
