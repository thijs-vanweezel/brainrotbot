"""Tests for the Step 7 upload-queue dedup (scan_ready must never surface an already-posted video)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from brainrotbot.ledger import append_entry
from brainrotbot.models import LedgerEntry
from brainrotbot.upload.queue import scan_ready


def _entry(variant_id: str, *, status: str = "thumbnail_done", tiktok_id=None) -> LedgerEntry:
    """Minimal entry keyed by variant_id with a (claimed) final video path filled in below."""
    return LedgerEntry(
        id=variant_id,  # use the variant id as the entry id too for a simple, unique key
        created_at=time.time(),
        status=status,
        source={"post_id": variant_id.split("_")[0], "variant_id": variant_id},
        text={"title": variant_id},
        assets={"final_video": None},
        upload={"tiktok_id": tiktok_id},
    )


def _settings(tmp_path):
    stories = tmp_path / "stories"
    stories.mkdir()
    return SimpleNamespace(stories_dir=stories, ledger_path=tmp_path / "ledger.jsonl")


def _write_story(settings, entry: LedgerEntry, tmp_path) -> None:
    """Write the per-story JSON plus a real (dummy) final-video file so scan_ready sees it on disk."""
    final = tmp_path / f"{entry.source['variant_id']}.mp4"
    final.write_bytes(b"\x00")
    entry.assets["final_video"] = str(final)
    (settings.stories_dir / f"{entry.source['variant_id']}.json").write_text(
        json.dumps(entry.to_dict(), ensure_ascii=False), encoding="utf-8"
    )


def test_ready_when_unposted(tmp_path):
    settings = _settings(tmp_path)
    _write_story(settings, _entry("abc_en"), tmp_path)
    assert [e.source["variant_id"] for e in scan_ready(settings)] == ["abc_en"]


def test_skips_when_story_json_marks_uploaded(tmp_path):
    settings = _settings(tmp_path)
    _write_story(settings, _entry("abc_en", status="upload_done", tiktok_id="123"), tmp_path)
    assert scan_ready(settings) == []


def test_skips_when_ledger_marks_uploaded_even_if_story_json_lost_status(tmp_path):
    """The story JSON forgot the upload (status reverted to thumbnail_done) but the ledger remembers.

    This is the case the user hit: cross-checking the ledger must still block the re-post.
    """
    settings = _settings(tmp_path)
    # Ledger says abc_en was posted...
    append_entry(settings.ledger_path, _entry("abc_en", status="upload_done", tiktok_id="123"))
    # ...but the on-disk story JSON looks un-posted again (lost/reverted/regenerated).
    _write_story(settings, _entry("abc_en", status="thumbnail_done"), tmp_path)
    assert scan_ready(settings) == []


def test_sibling_language_variant_not_blocked(tmp_path):
    """Posting abc_en must NOT block abc_fr -- different language variants are separate videos."""
    settings = _settings(tmp_path)
    append_entry(settings.ledger_path, _entry("abc_en", status="upload_done", tiktok_id="123"))
    _write_story(settings, _entry("abc_fr", status="thumbnail_done"), tmp_path)
    assert [e.source["variant_id"] for e in scan_ready(settings)] == ["abc_fr"]
