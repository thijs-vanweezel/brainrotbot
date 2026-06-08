"""Tests for the Step 7 upload-queue dedup (scan_ready must never surface an already-posted video)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from brainrotbot.ledger import append_entry
from brainrotbot.models import LedgerEntry
from brainrotbot.upload import queue as queue_mod
from brainrotbot.upload.queue import drain_upload_queue, reconcile_scheduled, scan_ready
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

def _drain_settings(tmp_path, *, schedule: bool = False, provider: str = "playwright"):
    """Full settings stub for drain_upload_queue (scan_ready fields + the bits the drain reads).

    Defaults to provider="playwright" so the Playwright-path tests below exercise upload()/upload_batch
    without the default Zernio branch; the Zernio tests pass provider="zernio". schedule=False is the
    live one-by-one path; schedule=True is the bulk-schedule path.
    """
    s = _settings(tmp_path)
    s.upload_opts = {"delete_after_upload": True, "headless": True,
                     "schedule": schedule, "provider": provider}
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


# --- bulk-schedule path -------------------------------------------------------------------------

class _FakeScheduler:
    """Drop-in for TikTokUploader on the bulk-schedule path: replays a scripted per-item result list."""

    def __init__(self, results, *, privacy="public", **_kw):
        self._results = results
        self.privacy = privacy
        self.batch_calls = 0
        self.items = None
        self.schedule_times = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def upload_batch(self, items, schedule_times):
        self.batch_calls += 1
        self.items = items
        self.schedule_times = schedule_times
        return self._results


def _ok(scheduled_for=1.0, cover_set=False):
    return {"ok": True, "scheduled_for": scheduled_for, "cover_set": cover_set, "error": None}


def test_schedule_marks_scheduled_keeps_media_and_no_resurface(tmp_path, monkeypatch):
    """Bulk path: scheduled videos become upload_scheduled, keep their media, and don't re-surface."""
    settings = _drain_settings(tmp_path, schedule=True)
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    _write_story(settings, _entry("b_en", status="thumbnail_done"), tmp_path)
    fake = _FakeScheduler([_ok(), _ok()])
    monkeypatch.setattr(queue_mod, "TikTokUploader", lambda **kw: fake)

    n = drain_upload_queue(settings)

    assert n == 2 and fake.batch_calls == 1  # one session, two videos
    assert _status(settings, "a_en") == "upload_scheduled"
    assert _status(settings, "b_en") == "upload_scheduled"
    assert _final_exists(tmp_path, "a_en") and _final_exists(tmp_path, "b_en")  # media KEPT until publish
    assert scan_ready(settings) == []  # upload_scheduled is in the skip set -> never re-scheduled
    # Schedule times are spread (b after a), not all the same instant.
    assert fake.schedule_times[1] > fake.schedule_times[0]


def test_schedule_failure_reverts_and_keeps_media(tmp_path, monkeypatch):
    """A per-item schedule failure reverts that one (retriable, media kept); the ok one is scheduled."""
    settings = _drain_settings(tmp_path, schedule=True)
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    _write_story(settings, _entry("b_en", status="thumbnail_done"), tmp_path)
    fake = _FakeScheduler([
        {"ok": False, "scheduled_for": None, "cover_set": False, "error": "Schedule option not found"},
        _ok(),
    ])
    monkeypatch.setattr(queue_mod, "TikTokUploader", lambda **kw: fake)

    n = drain_upload_queue(settings)

    assert n == 1
    assert _status(settings, "a_en") == "thumbnail_done"  # reverted -> retriable
    assert _status(settings, "b_en") == "upload_scheduled"
    assert _final_exists(tmp_path, "a_en")  # failed item keeps media
    assert [e.source["variant_id"] for e in scan_ready(settings)] == ["a_en"]  # only the failed one retries


def test_max_per_run_caps_the_batch(tmp_path, monkeypatch):
    """Only max_per_run videos are scheduled per drain; the rest stay queued for the next run."""
    settings = _drain_settings(tmp_path, schedule=True)
    settings.upload_opts["max_per_run"] = 1
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    _write_story(settings, _entry("b_en", status="thumbnail_done"), tmp_path)
    fake = _FakeScheduler([_ok()])  # only one item handed to the uploader
    monkeypatch.setattr(queue_mod, "TikTokUploader", lambda **kw: fake)

    n = drain_upload_queue(settings)

    assert n == 1 and len(fake.items) == 1  # capped to max_per_run
    assert [e.source["variant_id"] for e in scan_ready(settings)] == ["b_en"]  # the rest still queued


# --- reconcile ----------------------------------------------------------------------------------

class _FakeContentManager:
    """Drop-in for TikTokUploader on the reconcile path: returns scripted published videos."""

    def __init__(self, published, **_kw):
        self._published = published

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def published_videos(self):
        return self._published


def test_reconcile_finalizes_scheduled_and_cleans_media(tmp_path, monkeypatch):
    """A published scheduled post is matched to its URL, marked upload_done, and its media deleted."""
    settings = _drain_settings(tmp_path, schedule=True)
    entry = _entry("a_en", status="upload_scheduled")
    entry.upload["scheduled_for"] = 100.0
    _write_story(settings, entry, tmp_path)
    fake = _FakeContentManager([{"url": "https://www.tiktok.com/@x/video/9", "tiktok_id": "9"}])
    monkeypatch.setattr(queue_mod, "TikTokUploader", lambda **kw: fake)

    n = reconcile_scheduled(settings)

    assert n == 1
    assert _status(settings, "a_en") == "upload_done"
    rec = json.loads((settings.stories_dir / "a_en.json").read_text(encoding="utf-8"))
    assert rec["upload"]["tiktok_id"] == "9"
    assert not _final_exists(tmp_path, "a_en")  # delete_after_upload removed the media post-publish


def test_reconcile_noop_without_scheduled(tmp_path, monkeypatch):
    """Reconcile does nothing (and never opens a browser) when no posts await reconciliation."""
    settings = _drain_settings(tmp_path, schedule=True)
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    called = {"n": 0}
    monkeypatch.setattr(queue_mod, "TikTokUploader",
                        lambda **kw: called.__setitem__("n", called["n"] + 1))

    assert reconcile_scheduled(settings) == 0
    assert called["n"] == 0  # short-circuited before building an uploader


# --- Zernio provider (default) ------------------------------------------------------------------

class _FakeZernio:
    """Drop-in for ZernioClient: records scheduled posts and replays a fixed status payload."""

    def __init__(self, post_id="zp_1", status_payload=None, **_kw):
        self.post_id = post_id
        self.status_payload = status_payload or {}
        self.scheduled = []

    def schedule_video(self, url, caption, when, **_kw):
        self.scheduled.append((url, caption, when))
        return self.post_id

    def post_status(self, _post_id):
        return self.status_payload


def test_zernio_schedules_keeps_media_and_records_post_id(tmp_path, monkeypatch):
    """Default Zernio path: stash to Litterbox, POST to Zernio, mark upload_scheduled, keep media."""
    settings = _drain_settings(tmp_path, provider="zernio")
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    _write_story(settings, _entry("b_en", status="thumbnail_done"), tmp_path)
    fake = _FakeZernio(post_id="zp_x")
    monkeypatch.setattr(queue_mod, "ZernioClient", lambda *a, **k: fake)
    monkeypatch.setattr(queue_mod, "upload_to_litterbox", lambda p, **k: "https://litterbox/" + Path(p).name)

    n = drain_upload_queue(settings)

    assert n == 2 and len(fake.scheduled) == 2
    rec = json.loads((settings.stories_dir / "a_en.json").read_text(encoding="utf-8"))
    assert rec["status"] == "upload_scheduled"
    assert rec["upload"]["zernio_post_id"] == "zp_x" and rec["upload"]["provider"] == "zernio"
    assert _final_exists(tmp_path, "a_en") and _final_exists(tmp_path, "b_en")  # media KEPT until publish
    assert scan_ready(settings) == []  # scheduled -> skip set -> never re-scheduled
    assert fake.scheduled[1][2] > fake.scheduled[0][2]  # spread in time


def test_zernio_litterbox_failure_reverts_and_keeps_media(tmp_path, monkeypatch):
    """If the Litterbox upload (or Zernio POST) fails, the entry stays retriable with media kept."""
    settings = _drain_settings(tmp_path, provider="zernio")
    _write_story(settings, _entry("a_en", status="thumbnail_done"), tmp_path)
    monkeypatch.setattr(queue_mod, "ZernioClient", lambda *a, **k: _FakeZernio())

    def boom(_p, **_k):
        raise RuntimeError("litterbox down")
    monkeypatch.setattr(queue_mod, "upload_to_litterbox", boom)

    assert drain_upload_queue(settings) == 0
    assert _status(settings, "a_en") == "thumbnail_done"  # reverted, retriable
    assert _final_exists(tmp_path, "a_en")
    assert [e.source["variant_id"] for e in scan_ready(settings)] == ["a_en"]


def test_zernio_reconcile_published_writes_url_and_cleans(tmp_path, monkeypatch):
    """A published Zernio post -> live URL captured, upload_done, media deleted."""
    settings = _drain_settings(tmp_path, provider="zernio")
    entry = _entry("a_en", status="upload_scheduled")
    entry.upload.update(zernio_post_id="zp_x", scheduled_for=1.0)
    _write_story(settings, entry, tmp_path)
    payload = {"status": "published", "postUrl": "https://www.tiktok.com/@x/video/77"}
    monkeypatch.setattr(queue_mod, "ZernioClient", lambda *a, **k: _FakeZernio(status_payload=payload))

    n = reconcile_scheduled(settings)

    assert n == 1
    rec = json.loads((settings.stories_dir / "a_en.json").read_text(encoding="utf-8"))
    assert rec["status"] == "upload_done" and rec["upload"]["url"].endswith("/video/77")
    assert not _final_exists(tmp_path, "a_en")  # cleaned after confirmed publish


def test_zernio_reconcile_pending_leaves_scheduled(tmp_path, monkeypatch):
    """A still-pending Zernio post stays upload_scheduled with media kept (retried next run)."""
    settings = _drain_settings(tmp_path, provider="zernio")
    entry = _entry("a_en", status="upload_scheduled")
    entry.upload.update(zernio_post_id="zp_x", scheduled_for=1.0)
    _write_story(settings, entry, tmp_path)
    monkeypatch.setattr(queue_mod, "ZernioClient",
                        lambda *a, **k: _FakeZernio(status_payload={"status": "scheduled"}))

    assert reconcile_scheduled(settings) == 0
    assert _status(settings, "a_en") == "upload_scheduled"
    assert _final_exists(tmp_path, "a_en")
