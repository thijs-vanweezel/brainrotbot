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
import re
import time
from pathlib import Path

from ..ledger import append_entry, iter_entries
from ..models import LedgerEntry
from ..wiki.article import fetch_random_article
from .tiktok import PostFailedError, TikTokUploader, UploadRejectedError

# Statuses that mean an upload was already started or finished for this video; scan_ready never
# re-surfaces these, so we can't double-post. `upload_attempting` is the crash-safety breadcrumb
# written just before clicking Post -- if the process dies mid-upload it stays on disk and the next
# run skips the video (surfaced for manual review) instead of posting it a second time.
# `upload_discarded` is a dead Post button we deliberately gave up on (see PostFailedError handling).
_ATTEMPTED_STATUSES = ("upload_done", "upload_unconfirmed", "upload_attempting", "upload_discarded")


def _post_id(entry: LedgerEntry) -> str:
    # Filenames/JSON are keyed per language variant (Step 1.5 fan-out): `<post_id>_<lang>`.
    # `post_id` itself stays the base Reddit id (used only for dedup), so prefer `variant_id` here.
    return entry.source.get("variant_id") or entry.source.get("post_id") or entry.id


def _uploaded_keys(ledger_path) -> set[str]:
    """Per-video ids that the append-only ledger records as already posted/attempted.

    The per-story JSON is normally authoritative, but the ledger is the durable history of every
    upload, so cross-checking it means a video posted in an earlier run can never be re-posted even
    if its story JSON was lost, reverted, or regenerated. Keys are `_post_id` (the per-video
    variant_id) + the entry id -- NOT the base Reddit post_id, so uploading `<post>_en` never blocks
    its sibling `<post>_fr` (different language variants are deliberately separate videos).
    """
    keys: set[str] = set()
    for entry in iter_entries(ledger_path):
        if entry.upload.get("tiktok_id") or entry.status in _ATTEMPTED_STATUSES:
            keys.add(_post_id(entry))
            if entry.id:
                keys.add(entry.id)
    return keys


def scan_ready(settings) -> list[LedgerEntry]:
    """Ready = final video present on disk and the post has not been attempted yet.

    Reads the per-story JSON (updated in place by each step), which is the authoritative latest state
    for a post, AND cross-checks the append-only ledger: a video is skipped if its own JSON says it
    was attempted (status `upload_done`/`upload_unconfirmed` or a `tiktok_id`) OR if the ledger has it
    recorded as posted. Either signal alone blocks a re-post, so we never double-post; unconfirmed
    ones are surfaced for manual review instead. Sorted oldest-first.
    """
    stories_dir = settings.stories_dir
    if not stories_dir.is_dir():
        return []
    uploaded = _uploaded_keys(settings.ledger_path)
    ready: list[LedgerEntry] = []
    for path in sorted(stories_dir.glob("*.json")):
        try:
            entry = LedgerEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 -- skip an unreadable/partial story file
            continue
        if entry.upload.get("tiktok_id") or entry.status in _ATTEMPTED_STATUSES:
            continue  # already uploaded or already attempted (per this story file) -- don't re-post
        if _post_id(entry) in uploaded or entry.id in uploaded:
            continue  # the ledger already has this video as posted/attempted -- don't re-post
        final = entry.assets.get("final_video")
        if final and Path(final).is_file():
            ready.append(entry)
    ready.sort(key=lambda e: e.created_at)
    return ready


def _subreddit_hashtag(entry: LedgerEntry) -> str | None:
    """The story's own source subreddit as a hashtag (#AmItheAsshole), or None if unknown.

    Sanitised to the alphanumerics/underscore TikTok allows in a tag.
    """
    sub = re.sub(r"[^0-9A-Za-z_]", "", (entry.source.get("subreddit") or "").strip())
    return f"#{sub}" if sub else None


def _truncate(text: str, limit: int) -> str:
    """Trim to <= `limit` chars, cutting at a word boundary and appending an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    cut = text[: limit - 1].rstrip()           # leave room for the 1-char ellipsis
    sp = cut.rfind(" ")
    if sp > 0:
        cut = cut[:sp].rstrip()
    return cut + "…"


def _build_caption(
    entry: LedgerEntry,
    hashtags: list[str],
    *,
    article: dict | None,
    fallback_template: str,
    max_chars: int,
) -> str:
    """Caption = a (random, unrelated) Wikipedia intro + the hashtag block, capped at `max_chars`.

    The hashtag block drives reach, so it's built whole first; the body is then truncated at a
    word boundary to fit. Degrades to the post title (via `fallback_template`) when no article
    was fetched (e.g. Wikipedia outage).
    """
    body = (article or {}).get("extract") or fallback_template.format(title=entry.text.get("title", ""))
    tags = " ".join(hashtags)
    if not tags:
        return _truncate(body, max_chars)
    room = max_chars - len(tags) - 2           # reserve the tags + a blank-line separator
    if room <= 0:                              # pathological: tags alone exceed the cap -> tags only
        return tags[:max_chars].strip()
    body = _truncate(body, room)
    return f"{body}\n\n{tags}".strip() if body else tags


def _rewrite_story_file(settings, entry: LedgerEntry) -> None:
    """Persist the updated entry back to its per-story JSON (in-place, like pipeline._write_story_file)."""
    path = settings.stories_dir / f"{_post_id(entry)}.json"
    path.write_text(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_assets(settings, entry: LedgerEntry) -> list[str]:
    """Delete this post's heavy per-video media once it's safely uploaded; keep JSON + shared caches.

    Removes final / background / audio / subtitle / thumbnail files (regenerable, and large). Leaves the
    music MP3 (it lives in the shared, reused music_cache) and every *_cache dir untouched. The .ass is an
    intermediate already burned into the final video, so it goes too. Returns what was deleted.
    """
    deleted: list[str] = []
    for key in ("final_video", "background_video", "audio_path", "subtitles_path", "thumbnail_path"):
        p = entry.assets.get(key)
        if p and Path(p).is_file():
            try:
                Path(p).unlink()
                deleted.append(key)
            except OSError as exc:
                print(f"[brainrotbot]   (could not delete {key} for {_post_id(entry)}: {exc})")
    return deleted


def drain_upload_queue(settings, *, headless: bool | None = None, debug: bool = False) -> int:
    """Upload every ready video to TikTok in one browser session. Returns the number confirmed posted.

    Resilient: one video's failure logs a warning and never aborts the batch. **Confirm-before-commit**:
    a post counts as done only when a real video URL was captured -- then status=upload_done, the URL is
    recorded, and (delete_after_upload) the media is deleted. If Post was clicked but no URL came back, the
    entry is marked `upload_unconfirmed` and its media is KEPT (scan_ready then won't re-post it, so no
    duplicate). The updated entry is rewritten to the story JSON and re-appended to the append-only ledger
    (Step 8 dedupes by id, taking the latest line). `debug` dumps the Studio DOM at each milestone.
    """
    ready = scan_ready(settings)
    if not ready:
        print("[brainrotbot] Upload queue empty -- nothing to drain.")
        return 0

    opts = settings.upload_opts
    template = opts.get("caption_template", "{title}")           # fallback body if Wikipedia is down
    hashtags = [h for h in opts.get("hashtags", []) if h]        # fixed pool, same every post
    max_chars = int(opts.get("caption_max_chars", 2200))
    append_sub = bool(opts.get("append_subreddit_hashtag", True))
    wiki_enabled = bool(settings.wikipedia_opts.get("enabled", True))
    delete_after = bool(opts.get("delete_after_upload", True))
    headless = opts.get("headless", False) if headless is None else headless
    debug = debug or bool(opts.get("debug", False))

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
        completion_timeout_sec=float(opts.get("completion_timeout_sec", 300)),
        post_settle_sec=float(opts.get("post_settle_sec", 30)),
        debug=debug,
        debug_dir=settings.data_dir / "upload_debug",
    )
    with uploader:
        for entry in ready:
            pid = _post_id(entry)
            # Random, deliberately-unrelated Wikipedia intro as the description body (+ its category
            # recorded below for Step 8). Tags = the fixed pool plus this story's own subreddit.
            article = fetch_random_article(settings) if wiki_enabled else None
            entry_tags = list(hashtags)
            if append_sub:
                sub_tag = _subreddit_hashtag(entry)
                if sub_tag and sub_tag.lower() not in {t.lower() for t in entry_tags}:
                    entry_tags.append(sub_tag)
            caption = _build_caption(
                entry, entry_tags, article=article, fallback_template=template, max_chars=max_chars
            )
            cover = entry.assets.get("thumbnail_path")
            # Persist an in-flight marker BEFORE posting. The post is a live side-effect; if the
            # process is killed between the post and our success-write, this on-disk status is the
            # only thing that stops the next run re-posting the (still-on-disk) final. A normal
            # upload() exception means the post almost certainly didn't happen, so we revert the
            # marker below to keep retrying that genuine failure.
            prev_status = entry.status
            entry.status = "upload_attempting"
            _rewrite_story_file(settings, entry)
            try:
                meta = uploader.upload(
                    Path(entry.assets["final_video"]),
                    Path(cover) if cover and Path(cover).is_file() else None,
                    caption,
                )
                confirmed = bool(meta["url"] or meta["tiktok_id"])  # a captured URL = proof it posted
                entry.upload.update(
                    tiktok_id=meta["tiktok_id"],
                    url=meta["url"],
                    posted_at=meta["posted_at"],
                    caption=caption,
                    hashtags=entry_tags,
                    public=meta["public"],
                    captions_on=meta["captions_on"],
                    cover_set=meta["cover_set"],
                    content_check_off=meta.get("content_check_off"),
                )
                if article:
                    # Step 8 A/B signal: the (unrelated) Wikipedia article whose intro padded the caption.
                    entry.upload["wikipedia"] = {
                        "title": article["title"], "url": article["url"],
                        "category": article["category"], "categories": article["categories"],
                    }
                if confirmed:
                    entry.status = "upload_done"
                    if delete_after:
                        entry.upload["assets_deleted"] = _cleanup_assets(settings, entry)
                    posted += 1
                    print(f"[brainrotbot]   uploaded {pid} -> {meta['url']}")
                else:
                    # Post was clicked but never confirmed -- keep the media, don't auto-retry.
                    entry.status = "upload_unconfirmed"
                    print(f"[brainrotbot] WARNING: {pid} posted but URL not confirmed -- left as "
                          f"upload_unconfirmed, media kept. Check TikTok manually.")
                _rewrite_story_file(settings, entry)
                append_entry(settings.ledger_path, entry)
            except PostFailedError as exc:
                # Dead Post button (e.g. the daily check limit froze the form): the post definitely
                # didn't land, so discard this video rather than stall the batch. Mark it
                # upload_discarded so scan_ready won't re-surface it, and -- like a successful upload --
                # delete its heavy media when delete_after is on. Warn and continue with the next one.
                entry.status = "upload_discarded"
                entry.upload["discarded_reason"] = str(exc)
                if delete_after:
                    entry.upload["assets_deleted"] = _cleanup_assets(settings, entry)
                _rewrite_story_file(settings, entry)
                append_entry(settings.ledger_path, entry)
                print(f"[brainrotbot] WARNING: Post button did not work for {pid} -- discarded ({exc}).")
            except UploadRejectedError as exc:
                # TikTok positively rejected the post (it did NOT go through) -- typically the daily check
                # limit. We KNOW it didn't post, so revert to the prior status (retriable) and KEEP the
                # media: scan_ready will re-surface it on a future run (e.g. tomorrow, once the limit
                # resets). Don't append to the ledger (nothing was posted).
                entry.status = prev_status
                _rewrite_story_file(settings, entry)
                print(f"[brainrotbot] WARNING: post rejected for {pid} ({exc}) -- left retriable, media kept.")
                if exc.daily_limit:
                    # The daily cap blocks every remaining video today; stop now and retry next run.
                    print("[brainrotbot] Daily check limit reached -- aborting drain; "
                          f"{len(ready) - posted} video(s) stay queued for a future run.")
                    break
            except Exception as exc:  # noqa: BLE001 -- one bad upload must not abort the batch
                # Clean failure (not a hard crash): the post didn't go through, so clear the in-flight
                # marker and let the next run retry. A hard crash skips this and leaves the marker.
                entry.status = prev_status
                _rewrite_story_file(settings, entry)
                print(f"[brainrotbot] WARNING: upload failed for {pid}: {exc}")
    print(f"[brainrotbot] Drain complete: {posted}/{len(ready)} confirmed posted.")
    return posted
