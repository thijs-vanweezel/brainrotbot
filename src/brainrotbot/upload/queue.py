"""Step 7 queue drain: find finished-but-unuploaded videos on disk and push them to TikTok.

The "queue" is just the durable state on disk: any data/stories/<post_id>.json whose final video
exists and whose upload has no tiktok_id yet. Scanning disk (rather than only this run's in-memory
entries) means a crash or a separate `--upload-only` flush never strands a finished video, and leftovers
from earlier runs get swept up too.

`drain_upload_queue` opens ONE browser for the whole batch (cheaper + far less bot-risky than relaunching
per video), uploads each ready entry, writes the resulting URL back to the ledger + story file, and -- when
[upload].delete_after_upload is on -- deletes that post's heavy media (final/background/audio/thumbnail),
keeping only the JSON record and the shared caches for Step 8 analytics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..ledger import append_entry
from ..models import LedgerEntry
from .tiktok import TikTokUploader


def _post_id(entry: LedgerEntry) -> str:
    return entry.source.get("post_id", entry.id)


def scan_ready(settings) -> list[LedgerEntry]:
    """Ready = final video present on disk and not yet uploaded (upload.tiktok_id is None).

    Reads the per-story JSON (updated in place by each step), which is the authoritative latest state
    for a post. Sorted oldest-first so the queue drains FIFO.
    """
    stories_dir = settings.stories_dir
    if not stories_dir.is_dir():
        return []
    ready: list[LedgerEntry] = []
    for path in sorted(stories_dir.glob("*.json")):
        try:
            entry = LedgerEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 -- skip an unreadable/partial story file
            continue
        if entry.upload.get("tiktok_id"):
            continue  # already uploaded
        final = entry.assets.get("final_video")
        if final and Path(final).is_file():
            ready.append(entry)
    ready.sort(key=lambda e: e.created_at)
    return ready


def _build_caption(entry: LedgerEntry, template: str, hashtags: list[str]) -> str:
    """caption_template (default '{title}') + space-joined hashtags."""
    base = template.format(title=entry.text.get("title", ""))
    tags = " ".join(hashtags)
    return f"{base} {tags}".strip()


def _rewrite_story_file(settings, entry: LedgerEntry) -> None:
    """Persist the updated entry back to its per-story JSON (in-place, like pipeline._write_story_file)."""
    path = settings.stories_dir / f"{_post_id(entry)}.json"
    path.write_text(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_assets(settings, entry: LedgerEntry) -> list[str]:
    """Delete this post's heavy per-video media once it's safely uploaded; keep JSON + shared caches.

    Removes final / background / audio / thumbnail files (regenerable, and large). Leaves the music MP3
    (it lives in the shared, reused music_cache) and every *_cache dir untouched. Returns what was deleted.
    """
    deleted: list[str] = []
    for key in ("final_video", "background_video", "audio_path", "thumbnail_path"):
        p = entry.assets.get(key)
        if p and Path(p).is_file():
            try:
                Path(p).unlink()
                deleted.append(key)
            except OSError as exc:
                print(f"[brainrotbot]   (could not delete {key} for {_post_id(entry)}: {exc})")
    return deleted


def drain_upload_queue(settings, *, headless: bool | None = None) -> int:
    """Upload every ready video to TikTok in one browser session. Returns the number posted.

    Resilient: one video's failure logs a warning and leaves it in the queue (retried next drain) without
    aborting the batch. A successful post updates entry.upload + status, rewrites the story JSON, appends
    the updated entry to the ledger (append-only, so Step 8 must dedupe by id taking the latest line), and
    optionally deletes the post's media.
    """
    ready = scan_ready(settings)
    if not ready:
        print("[brainrotbot] Upload queue empty -- nothing to drain.")
        return 0

    opts = settings.upload_opts
    template = opts.get("caption_template", "{title}")
    hashtags = [h for h in opts.get("hashtags", []) if h]
    delete_after = bool(opts.get("delete_after_upload", True))
    headless = opts.get("headless", False) if headless is None else headless

    print(f"[brainrotbot] Draining upload queue: {len(ready)} video(s) ready for TikTok.")
    posted = 0
    uploader = TikTokUploader(
        session_dir=settings.tiktok_session_dir,
        browser=opts.get("browser", "chromium"),
        cookies_file=settings.tiktok_cookies_file,
        user_agent=opts.get("user_agent", ""),
        upload_url=opts.get("upload_url", "https://www.tiktok.com/tiktokstudio/upload"),
        headless=headless,
        privacy=opts.get("privacy", "public"),
        subtitles=bool(opts.get("subtitles", True)),
        set_cover=bool(opts.get("set_cover", True)),
        nav_timeout_sec=float(opts.get("nav_timeout_sec", 120)),
    )
    with uploader:
        for entry in ready:
            pid = _post_id(entry)
            caption = _build_caption(entry, template, hashtags)
            cover = entry.assets.get("thumbnail_path")
            try:
                meta = uploader.upload(
                    Path(entry.assets["final_video"]),
                    Path(cover) if cover and Path(cover).is_file() else None,
                    caption,
                )
                entry.upload.update(
                    tiktok_id=meta["tiktok_id"],
                    url=meta["url"],
                    posted_at=meta["posted_at"],
                    caption=caption,
                    hashtags=hashtags,
                    public=meta["public"],
                    captions_on=meta["captions_on"],
                    cover_set=meta["cover_set"],
                )
                entry.status = "upload_done"
                if delete_after:
                    entry.upload["assets_deleted"] = _cleanup_assets(settings, entry)
                _rewrite_story_file(settings, entry)
                append_entry(settings.ledger_path, entry)
                posted += 1
                print(f"[brainrotbot]   uploaded {pid} -> {meta['url'] or '(url not captured)'}")
            except Exception as exc:  # noqa: BLE001 -- one bad upload must not abort the batch
                print(f"[brainrotbot] WARNING: upload failed for {pid}: {exc}")
    print(f"[brainrotbot] Drain complete: {posted}/{len(ready)} uploaded.")
    return posted
