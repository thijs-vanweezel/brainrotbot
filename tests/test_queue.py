"""Tests for the Step 7 upload-queue dedup (scan_ready must never surface an already-posted video)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from brainrotbot.ledger import append_entry
from brainrotbot.models import LedgerEntry
from brainrotbot.upload import queue as queue_mod
from brainrotbot.upload.queue import drain_upload_queue, scan_ready
from brainrotbot.upload.tiktok import UploadRejectedError


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


def test_skips_in_flight_marker_after_crash(tmp_path):
    """A crash mid-upload leaves status=upload_attempting on disk; the next run must not re-post it."""
    settings = _settings(tmp_path)
    _write_story(settings, _entry("abc_en", status="upload_attempting"), tmp_path)
    assert scan_ready(settings) == []


def test_sibling_language_variant_not_blocked(tmp_path):
    """Posting abc_en must NOT block abc_fr -- different language variants are separate videos."""
    settings = _settings(tmp_path)
    append_entry(settings.ledger_path, _entry("abc_en", status="upload_done", tiktok_id="123"))
    _write_story(settings, _entry("abc_fr", status="thumbnail_done"), tmp_path)
    assert [e.source["variant_id"] for e in scan_ready(settings)] == ["abc_fr"]


# --- drain rejection handling -------------------------------------------------------------------

def _drain_settings(tmp_path):
    """Full settings stub for drain_upload_queue (scan_ready fields + the bits the drain reads)."""
    s = _settings(tmp_path)
    s.upload_opts = {"delete_after_upload": True, "headless": True}
    s.wikipedia_opts = {"enabled": False}  # keep the drain offline/deterministic (no real Wikipedia call)
    s.tiktok_session_dir = tmp_path / "profile"
    s.tiktok_cookies_file = ""
    s.data_dir = tmp_path
    return s


class _FakeUploader:
    """Drop-in for TikTokUploader: a context manager whose upload() replays a scripted outcome list."""

    def __init__(self, outcomes, **_kw):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def upload(self, *_a, **_kw):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _final_exists(tmp_path, variant_id: str) -> bool:
    return (tmp_path / f"{variant_id}.mp4").is_file()


def _status(settings, variant_id: str) -> str:
    return json.loads((settings.stories_dir / f"{variant_id}.json").read_text(encoding="utf-8"))["status"]


def test_daily_limit_reverts_and_aborts_batch(tmp_path, monkeypatch):
    """A daily-limit rejection: revert the video (retriable, media kept) AND stop the whole batch."""
    settings = _drain_settings(tmp_path)
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    _write_story(settings, _entry("b_en", status="thumbnail_done"), tmp_path)
    fake = _FakeUploader([UploadRejectedError("limit", daily_limit=True)])
    monkeypatch.setattr(queue_mod, "TikTokUploader", lambda **kw: fake)

    posted = drain_upload_queue(settings)

    assert posted == 0
    assert fake.calls == 1  # batch aborted after the first (limit) rejection -- b_en never attempted
    assert _status(settings, "a_en") == "thumbnail_done"  # reverted to prior status (retriable)
    assert _status(settings, "b_en") == "thumbnail_done"  # untouched
    assert _final_exists(tmp_path, "a_en") and _final_exists(tmp_path, "b_en")  # media kept for retry
    assert scan_ready(settings)  # both still surface on the next run


def test_non_limit_rejection_reverts_and_continues(tmp_path, monkeypatch):
    """A non-limit rejection reverts that one video but the batch continues to the next."""
    settings = _drain_settings(tmp_path)
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    _write_story(settings, _entry("b_en", status="thumbnail_done"), tmp_path)
    ok = {"url": "https://www.tiktok.com/@x/video/9", "tiktok_id": "9", "posted_at": 1.0,
          "public": True, "captions_on": True, "cover_set": False}
    fake = _FakeUploader([UploadRejectedError("oops", daily_limit=False), ok])
    monkeypatch.setattr(queue_mod, "TikTokUploader", lambda **kw: fake)

    posted = drain_upload_queue(settings)

    assert posted == 1 and fake.calls == 2  # a_en rejected, b_en posted
    assert _status(settings, "a_en") == "thumbnail_done"  # reverted, retriable
    assert _final_exists(tmp_path, "a_en")  # rejected video keeps its media
    assert _status(settings, "b_en") == "upload_done"
