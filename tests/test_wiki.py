"""Tests for the Step 7 Wikipedia caption: article parsing + caption assembly/truncation."""

from __future__ import annotations

import time

from brainrotbot.models import LedgerEntry
from brainrotbot.upload.queue import _build_caption, _subreddit_hashtag, _truncate
from brainrotbot.wiki.article import _parse_response

# A captured-shape MediaWiki response (the real fields we read), incl. one hidden-cat-free page.
_API_JSON = {
    "query": {
        "pages": {
            "4581613": {
                "pageid": 4581613,
                "title": "History of Gmail",
                "extract": "The public history of Gmail dates back to 2004. Gmail is a webmail service.",
                "categories": [
                    {"ns": 14, "title": "Category:2004 software"},
                    {"ns": 14, "title": "Category:Gmail"},
                    {"ns": 14, "title": "Category:History of the Internet"},
                ],
            }
        }
    }
}


def _entry(subreddit: str = "tifu", title: str = "My story") -> LedgerEntry:
    return LedgerEntry(
        id="x", created_at=time.time(), status="thumbnail_done",
        source={"subreddit": subreddit}, text={"title": title},
    )


# --- article parsing ----------------------------------------------------------------------------

def test_parse_response_extracts_intro_and_strips_category_prefix():
    art = _parse_response(_API_JSON, "en")
    assert art["title"] == "History of Gmail"
    assert art["extract"].startswith("The public history of Gmail")
    assert art["categories"] == ["2004 software", "Gmail", "History of the Internet"]
    assert art["category"] == "2004 software"               # primary = first visible category
    assert art["url"] == "https://en.wikipedia.org/wiki/History_of_Gmail"


def test_parse_response_none_when_no_extract():
    empty = {"query": {"pages": {"1": {"title": "T", "extract": ""}}}}
    assert _parse_response(empty) is None
    assert _parse_response({}) is None


def test_parse_response_handles_missing_categories():
    no_cats = {"query": {"pages": {"1": {"title": "T", "extract": "Body."}}}}
    art = _parse_response(no_cats)
    assert art["categories"] == [] and art["category"] is None


# --- subreddit hashtag --------------------------------------------------------------------------

def test_subreddit_hashtag_sanitises():
    assert _subreddit_hashtag(_entry("AmItheAsshole")) == "#AmItheAsshole"
    assert _subreddit_hashtag(_entry("a b!c")) == "#abc"           # non-alnum stripped
    assert _subreddit_hashtag(_entry("")) is None


# --- truncation ---------------------------------------------------------------------------------

def test_truncate_cuts_at_word_boundary_with_ellipsis():
    out = _truncate("one two three four", 12)
    assert len(out) <= 12
    assert out.endswith("…") and " " not in out.rstrip("…")[-1:]  # no dangling partial word start
    assert out == "one two…"


def test_truncate_passthrough_when_short():
    assert _truncate("short", 100) == "short"


# --- caption assembly ---------------------------------------------------------------------------

def test_build_caption_uses_article_body_and_appends_tags():
    art = _parse_response(_API_JSON, "en")
    cap = _build_caption(_entry(), ["#fyp", "#tifu"], article=art,
                         fallback_template="{title}", max_chars=2200)
    assert cap.startswith("The public history of Gmail")
    assert cap.endswith("#fyp #tifu")


def test_build_caption_falls_back_to_title_without_article():
    cap = _build_caption(_entry(title="Hello"), ["#fyp"], article=None,
                         fallback_template="{title}", max_chars=2200)
    assert cap.startswith("Hello") and cap.endswith("#fyp")


def test_build_caption_respects_char_limit_and_keeps_tags_whole():
    art = {"extract": "word " * 200}                      # ~1000 chars of body
    tags = ["#fyp", "#viral", "#tifu"]
    cap = _build_caption(_entry(), tags, article=art, fallback_template="{title}", max_chars=80)
    assert len(cap) <= 80
    assert cap.endswith("#fyp #viral #tifu")              # the whole tag block survived
