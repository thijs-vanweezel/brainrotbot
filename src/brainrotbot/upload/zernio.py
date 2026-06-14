"""Step 7 uploader via Zernio's REST API (the default), instead of driving TikTok Studio with Playwright.

Why: driving the web uploader bot-flagged the account (frozen "Content Check Lite" + "Something went
wrong"). Zernio posts through TikTok's official Content Posting API using THEIR audited app, so posts go
out PUBLIC and scheduled with no browser automation and no Content Check Lite gate -- which removes both
the flagging risk and the brittle DOM-pinning. Zernio PULLS the video from a public URL (it does not
accept file uploads), so we first stash each final mp4 on Litterbox (catbox's 72h temporary host) and
hand Zernio that URL; Litterbox auto-expires it, covering our same-day scheduling window.

Auth: ZERNIO_API_KEY (Bearer), from .env. Account id: [upload].zernio_account_id, or auto-resolved via
GET /v1/profiles when exactly one TikTok account is connected. Only needs `requests` (a core dep) --
no Playwright, no extra. See [[brainrotbot-upload]].
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

ZERNIO_BASE = "https://zernio.com/api/v1"
LITTERBOX_URL = "https://litterbox.catbox.moe/resources/internals/api.php"
_TT_VIDEO_RE = re.compile(r"https?://[^\s\"']*tiktok\.com/[^\s\"']*?/video/\d+", re.I)


class ZernioError(RuntimeError):
    """A Zernio API call (or the Litterbox upload that feeds it) failed."""


def upload_to_litterbox(path: Path, *, expiry: str = "72h", timeout: float = 600.0) -> str:
    """Upload a file to Litterbox and return its public URL (auto-expires after `expiry`).

    Litterbox is catbox's temporary host: a single multipart POST, no auth/account, returns the URL as
    plaintext. 72h comfortably covers a same-day schedule window. Raises ZernioError on failure.
    """
    path = Path(path)
    with open(path, "rb") as f:
        resp = requests.post(
            LITTERBOX_URL,
            data={"reqtype": "fileupload", "time": expiry},
            files={"fileToUpload": (path.name, f, "video/mp4")},
            timeout=timeout,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise ZernioError(f"Litterbox upload did not return a URL: {url!r}")
    return url


def find_tiktok_url(obj) -> str | None:
    """Recursively dig a tiktok.com/.../video/<id> URL out of an arbitrary status payload."""
    if isinstance(obj, str):
        m = _TT_VIDEO_RE.search(obj)
        return m.group(0) if m else None
    if isinstance(obj, dict):
        for v in obj.values():
            if (u := find_tiktok_url(v)):
                return u
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if (u := find_tiktok_url(v)):
                return u
    return None


def classify_status(payload: dict) -> str:
    """Map Zernio's post-status payload to one of: published | failed | pending.

    The exact schema isn't documented, so this is intentionally lenient: any 'status'/'state' string
    found anywhere is matched against known terminal words; a live TikTok URL is treated as published.
    """
    if find_tiktok_url(payload):
        return "published"
    text = " ".join(_status_strings(payload)).lower()
    if re.search(r"\b(failed|error|rejected|cancell?ed)\b", text):
        return "failed"
    if re.search(r"\b(published|posted|completed|success|live)\b", text):
        return "published"
    return "pending"  # scheduled / processing / queued -> check again later


def _status_strings(obj) -> list[str]:
    """Collect values of any 'status'/'state' keys (at any depth) for terminal-state matching."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and re.search(r"status|state", str(k), re.I):
                out.append(v)
            else:
                out.extend(_status_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_status_strings(v))
    return out


class ZernioClient:
    """Thin wrapper over Zernio's REST API for scheduling TikTok video posts + reading their status."""

    def __init__(self, api_key: str, account_id: str = "", *, timezone: str = "Europe/Amsterdam",
                 base_url: str = ZERNIO_BASE, timeout: float = 60.0):
        if not api_key:
            raise ZernioError("ZERNIO_API_KEY is not set -- put it in .env (or [upload].ZERNIO_API_KEY)")
        self.api_key = api_key
        self.account_id = account_id
        self.timezone = timezone
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def resolve_account_id(self) -> str:
        """Return the configured accountId, or auto-pick the single connected, enabled TikTok account.

        Zernio's GET /v1/accounts returns one entry per connected social account; the id used in the
        posts payload is that account's `_id` (platform="tiktok"). Raises ZernioError if none/ambiguous
        so the caller can tell the user to set [upload].zernio_account_id explicitly.
        """
        if self.account_id:
            return self.account_id
        r = requests.get(f"{self.base_url}/accounts", headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        accounts = data if isinstance(data, list) else (data.get("accounts") or data.get("data") or [])
        tiktok = [a for a in accounts
                  if str(a.get("platform", "")).lower() == "tiktok" and a.get("enabled", True)]
        if len(tiktok) == 1:
            self.account_id = str(tiktok[0].get("_id") or tiktok[0].get("accountId") or tiktok[0].get("id") or "")
        if not self.account_id:
            raise ZernioError(
                f"could not resolve a single TikTok accountId from /accounts ({len(tiktok)} found); "
                "set [upload].zernio_account_id explicitly")
        return self.account_id

    def schedule_video(self, video_url: str, caption: str, when, *,
                       cover_url: str | None = None, privacy: str = "PUBLIC_TO_EVERYONE") -> str:
        """Schedule one TikTok video; return Zernio's post id. `when` is a naive local-time datetime.

        `cover_url` (a public image URL) becomes `tiktokSettings.video_cover_image_url` -- TikTok's custom
        cover; when omitted TikTok falls back to a frame from the video.
        """
        tiktok_settings = {
            "privacy_level": privacy,
            "allow_comment": True, "allow_duet": True, "allow_stitch": True,
            # Both flags are mandatory legal acknowledgements for the Content Posting API.
            "content_preview_confirmed": True, "express_consent_given": True,
        }
        if cover_url:
            tiktok_settings["video_cover_image_url"] = cover_url  # custom cover (our Step 6 thumbnail)
        body = {
            "content": caption,
            "mediaItems": [{"type": "video", "url": video_url}],
            "platforms": [{"platform": "tiktok", "accountId": self.resolve_account_id()}],
            "scheduledFor": when.strftime("%Y-%m-%dT%H:%M:%S"),
            "timezone": self.timezone,
            "tiktokSettings": tiktok_settings,
        }
        r = requests.post(f"{self.base_url}/posts", headers=self._headers, json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise ZernioError(f"Zernio POST /posts failed ({r.status_code}): {r.text[:300]}")
        data = r.json()
        # Zernio nests the created post under {"post": {...}} and keys its id as `_id` (Mongo-style);
        # accept the flat shapes too for safety.
        post = data["post"] if isinstance(data.get("post"), dict) else data
        post_id = post.get("_id") or post.get("id") or data.get("id") or data.get("postId")
        if not post_id:
            raise ZernioError(f"Zernio /posts response had no post id: {str(data)[:300]}")
        return str(post_id)

    def post_status(self, post_id: str) -> dict:
        """GET /v1/posts/{id} -- the raw status payload (classify with classify_status/find_tiktok_url)."""
        r = requests.get(f"{self.base_url}/posts/{post_id}", headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
