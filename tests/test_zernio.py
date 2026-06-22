"""Tests for the Zernio uploader client (Step 7 default path) -- pure helpers + payload, requests mocked."""

from __future__ import annotations

import pytest

from brainrotbot.upload import zernio as z
from brainrotbot.upload.zernio import (
    ZernioClient, ZernioError, ZernioNotFound, classify_status, find_tiktok_url,
)


class _Resp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, *, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --- pure helpers -------------------------------------------------------------------------------

def test_find_tiktok_url_digs_through_nested_payload():
    payload = {"data": {"platforms": [{"name": "tiktok",
               "postUrl": "https://www.tiktok.com/@user/video/12345"}]}}
    assert find_tiktok_url(payload) == "https://www.tiktok.com/@user/video/12345"


def test_find_tiktok_url_none_when_absent():
    assert find_tiktok_url({"status": "scheduled", "url": "https://example.com/x"}) is None


@pytest.mark.parametrize("payload,expected", [
    ({"postUrl": "https://www.tiktok.com/@u/video/9"}, "published"),  # a live URL = published
    ({"status": "published"}, "published"),
    ({"state": "completed"}, "published"),
    ({"status": "failed"}, "failed"),
    ({"status": "error", "reason": "x"}, "failed"),
    ({"status": "rejected"}, "failed"),
    ({"status": "deleted"}, "deleted"),       # user removed it -> deleted (checked before failed)
    ({"status": "cancelled"}, "deleted"),
    ({"state": "removed"}, "deleted"),
    ({"status": "scheduled"}, "pending"),
    ({"status": "processing"}, "pending"),
])
def test_classify_status(payload, expected):
    assert classify_status(payload) == expected


# --- client -------------------------------------------------------------------------------------

def test_client_requires_api_key():
    with pytest.raises(ZernioError):
        ZernioClient("")


def test_schedule_video_builds_payload_and_returns_id(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _Resp(json_data={"id": "zp_123"})

    monkeypatch.setattr(z.requests, "post", fake_post)
    client = ZernioClient("sk_test", account_id="acc_1", timezone="Europe/Amsterdam")
    import datetime as dt
    pid = client.schedule_video("https://litterbox/x.mp4", "hello #fyp",
                                dt.datetime(2026, 6, 8, 15, 30, 0))

    assert pid == "zp_123"
    assert captured["url"].endswith("/api/v1/posts")
    assert captured["headers"]["Authorization"] == "Bearer sk_test"
    body = captured["body"]
    assert body["content"] == "hello #fyp"
    assert body["mediaItems"] == [{"type": "video", "url": "https://litterbox/x.mp4"}]
    assert body["platforms"] == [{"platform": "tiktok", "accountId": "acc_1"}]
    assert body["scheduledFor"] == "2026-06-08T15:30:00"  # naive ISO, no tz suffix
    assert body["timezone"] == "Europe/Amsterdam"
    assert body["tiktokSettings"]["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert body["tiktokSettings"]["content_preview_confirmed"] is True
    assert body["tiktokSettings"]["express_consent_given"] is True


def test_schedule_video_includes_cover_when_given(monkeypatch):
    captured = {}
    monkeypatch.setattr(z.requests, "post",
                        lambda url, headers=None, json=None, timeout=None:
                        captured.update(body=json) or _Resp(json_data={"id": "zp_1"}))
    import datetime as dt
    ZernioClient("sk_test", account_id="acc_1").schedule_video(
        "https://x/v.mp4", "c", dt.datetime(2026, 1, 1, 0, 0, 0),
        cover_url="https://x/cover.png")
    assert captured["body"]["tiktokSettings"]["video_cover_image_url"] == "https://x/cover.png"


def test_schedule_video_omits_cover_when_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(z.requests, "post",
                        lambda url, headers=None, json=None, timeout=None:
                        captured.update(body=json) or _Resp(json_data={"id": "zp_1"}))
    import datetime as dt
    ZernioClient("sk_test", account_id="acc_1").schedule_video(
        "https://x/v.mp4", "c", dt.datetime(2026, 1, 1, 0, 0, 0))
    assert "video_cover_image_url" not in captured["body"]["tiktokSettings"]


def test_schedule_video_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(z.requests, "post",
                        lambda *a, **k: _Resp(status_code=403, text="forbidden"))
    client = ZernioClient("sk_test", account_id="acc_1")
    import datetime as dt
    with pytest.raises(ZernioError):
        client.schedule_video("https://x/x.mp4", "c", dt.datetime(2026, 1, 1, 0, 0, 0))


def test_schedule_video_extracts_nested_mongo_id(monkeypatch):
    # Zernio actually returns {"post": {"_id": ...}} -- the id must be read from there.
    monkeypatch.setattr(z.requests, "post",
                        lambda *a, **k: _Resp(json_data={"post": {"_id": "6a26d362", "content": "x"}}))
    import datetime as dt
    pid = ZernioClient("sk_test", account_id="acc_1").schedule_video(
        "https://x/v.mp4", "c", dt.datetime(2026, 1, 1, 0, 0, 0))
    assert pid == "6a26d362"


def test_schedule_video_raises_when_no_post_id(monkeypatch):
    monkeypatch.setattr(z.requests, "post", lambda *a, **k: _Resp(json_data={"ok": True}))
    client = ZernioClient("sk_test", account_id="acc_1")
    import datetime as dt
    with pytest.raises(ZernioError):
        client.schedule_video("https://x/x.mp4", "c", dt.datetime(2026, 1, 1, 0, 0, 0))


def test_resolve_account_id_auto_picks_single_tiktok(monkeypatch):
    monkeypatch.setattr(z.requests, "get", lambda *a, **k: _Resp(
        json_data={"accounts": [{"_id": "acc_99", "platform": "tiktok", "enabled": True},
                                 {"_id": "acc_ig", "platform": "instagram", "enabled": True}]}))
    client = ZernioClient("sk_test")  # no account_id configured
    assert client.resolve_account_id() == "acc_99"


def test_resolve_account_id_raises_when_ambiguous(monkeypatch):
    monkeypatch.setattr(z.requests, "get", lambda *a, **k: _Resp(
        json_data={"accounts": [{"_id": "a", "platform": "tiktok", "enabled": True},
                                 {"_id": "b", "platform": "tiktok", "enabled": True}]}))
    with pytest.raises(ZernioError):
        ZernioClient("sk_test").resolve_account_id()


def test_post_status_raises_not_found_on_404(monkeypatch):
    # A deleted post -> 404 -> ZernioNotFound (so reconcile cleans it up, not retries forever).
    monkeypatch.setattr(z.requests, "get", lambda *a, **k: _Resp(status_code=404, text="not found"))
    with pytest.raises(ZernioNotFound):
        ZernioClient("sk_test", account_id="acc_1").post_status("zp_gone")


def test_post_status_raises_generic_error_on_500(monkeypatch):
    # A transient server error stays a generic ZernioError (NOT ZernioNotFound) -> left retriable.
    monkeypatch.setattr(z.requests, "get", lambda *a, **k: _Resp(status_code=500, text="boom"))
    with pytest.raises(ZernioError) as exc:
        ZernioClient("sk_test", account_id="acc_1").post_status("zp_1")
    assert not isinstance(exc.value, ZernioNotFound)


def test_upload_to_litterbox_returns_url(monkeypatch, tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"\x00\x01")
    monkeypatch.setattr(z.requests, "post",
                        lambda *a, **k: _Resp(text="https://litter.box/abc.mp4\n"))
    assert z.upload_to_litterbox(f) == "https://litter.box/abc.mp4"


def test_upload_to_litterbox_raises_on_non_url(monkeypatch, tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"\x00")
    monkeypatch.setattr(z.requests, "post", lambda *a, **k: _Resp(text="ERROR: too big"))
    with pytest.raises(ZernioError):
        z.upload_to_litterbox(f)
