"""brainrotbot pipeline orchestrator -- the single entry point for the whole bot.

Currently implements Step 1 (Reddit -> select -> clean -> filter -> persist). Later
steps (TTS, background video, editing, upload, analysis) plug into `run()` and consume
the ledger entries produced here, so the launch command never changes.

Run via the standalone launcher (run.bat) or directly:
    python -m brainrotbot.pipeline [--top-k N] [--settings PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .ledger import append_entry, existing_post_ids
from .models import LedgerEntry, Story
from .reddit.fetch import fetch_candidates
from .reddit.select import select_stories
from .text.clean import clean_text
from .text.filter_words import filter_banned_words
from .edit.compose import VideoEditor
from .music.ncs import TrackMeta, discover_instrumental_tracks, download_track, pick_track
from .tts.synthesize import KokoroSynthesizer, pick_voice
from .video.background import BackgroundVideoMaker, pick_source


def run(
    settings_path=None,
    top_k: int | None = None,
    skip_tts: bool = False,
    skip_video: bool = False,
    skip_edit: bool = False,
    skip_music: bool = False,
) -> list[LedgerEntry]:
    settings = load_settings(settings_path)

    selection = dict(settings.selection)
    if top_k is not None:
        selection["top_k"] = top_k

    print(f"[brainrotbot] Fetching from r/{', r/'.join(settings.subreddits)} via RSS ...")

    candidates = fetch_candidates(settings)
    print(f"[brainrotbot] {len(candidates)} candidate self-posts pulled.")

    seen = existing_post_ids(settings.ledger_path)
    selected = select_stories(candidates, selection, seen_ids=seen)
    print(f"[brainrotbot] {len(selected)} stories selected (after filters + dedup).")

    # Build one synthesizer for the whole run; the model loads lazily on first use.
    tts_opts = settings.tts_opts
    synth = None if skip_tts else KokoroSynthesizer(
        device=tts_opts.get("device", "auto"),
        sample_rate=tts_opts.get("sample_rate", 24000),
    )

    # Build the background-video maker once; sources are downloaded+cached lazily on first use.
    video_opts = settings.video_opts
    maker = None if skip_video else BackgroundVideoMaker(
        cache_dir=settings.video_cache_dir,
        width=video_opts.get("width", 1080),
        height=video_opts.get("height", 1920),
        fps=video_opts.get("fps", 30),
        crf=video_opts.get("crf", 23),
        preset=video_opts.get("preset", "veryfast"),
        max_source_height=video_opts.get("max_source_height", 1080),
        intro_skip_sec=video_opts.get("intro_skip_sec", 5.0),
        cookies_from_browser=video_opts.get("cookies_from_browser", ""),
        cookies_file=settings.video_cookies_file,
    )

    # Build the editor once; geometry is reused from [video] so the outro matches the background.
    edit_opts = settings.edit_opts
    editor = None if skip_edit else VideoEditor(
        width=video_opts.get("width", 1080),
        height=video_opts.get("height", 1920),
        fps=video_opts.get("fps", 30),
        crf=edit_opts.get("crf", 23),
        preset=edit_opts.get("preset", "veryfast"),
        outro_file=settings.edit_outro_file,
        outro_duration_sec=edit_opts.get("outro_duration_sec", 4.0),
        music_volume_db=edit_opts.get("music_volume_db", -15.0),
        music_duck=edit_opts.get("music_duck", True),
    )

    # Scrape the NCS instrumental catalogue once per run (cached by music/ncs.py on disk).
    # A scrape failure just disables music for this run -- it must never block the pipeline.
    music_on = (not skip_music) and edit_opts.get("music_enabled", True)
    music_catalogue: list[TrackMeta] = []
    if music_on:
        try:
            music_catalogue = discover_instrumental_tracks(
                settings.music_cache_dir,
                ttl_days=int(edit_opts.get("music_catalogue_ttl_days", 7)),
                num_pages=int(edit_opts.get("music_catalogue_pages", 3)),
            )
            print(f"[brainrotbot] NCS catalogue: {len(music_catalogue)} instrumental tracks.")
        except Exception as exc:  # noqa: BLE001 -- music is a polish; never abort over it
            print(f"[brainrotbot] WARNING: NCS catalogue unavailable, continuing without music: {exc}")

    settings.stories_dir.mkdir(parents=True, exist_ok=True)
    entries: list[LedgerEntry] = []
    for index, story in enumerate(selected):
        entry = _process_story(story, settings)
        if synth is not None:
            _add_audio(entry, story, synth, settings, index)
        if maker is not None:
            _add_background_video(entry, story, maker, settings, index)
        if music_catalogue:
            _add_music_bed(entry, story, music_catalogue, settings)
        if editor is not None:
            _add_final_video(entry, story, editor, settings)
        _write_story_file(settings, story, entry)
        append_entry(settings.ledger_path, entry)
        entries.append(entry)

    _print_summary(entries)

    # Future steps consume `entries` here and fill the ledger's assets/upload/metrics:
    #   2. text-to-speech    -> entry.assets["audio_path"]        (DONE)
    #   3. background video  -> entry.assets["background_video"]  (DONE)
    #   4. editing           -> entry.assets["final_video"]       (DONE)
    #   5. background music  -> entry.assets["music_path"] + mix  (DONE -- mixed in Step 4 pass)
    #   6. upload to TikTok  -> entry.upload[...]
    #   7. analysis          -> entry.metrics[...] + entry.content_analysis[...]
    return entries


def _process_story(story: Story, settings) -> LedgerEntry:
    text_opts = settings.text_opts
    cleaned = clean_text(
        story.title,
        story.raw_body,
        prepend_title=text_opts.get("prepend_title", True),
        strip_edits=text_opts.get("strip_edits", True),
        strip_tldr=text_opts.get("strip_tldr", True),
    )
    cleaned, replacements = filter_banned_words(cleaned, settings.banned_words_path)
    return LedgerEntry.from_story(story, cleaned, replacements)


def _add_audio(entry: LedgerEntry, story: Story, synth: KokoroSynthesizer, settings, index: int) -> None:
    """Narrate the cleaned story to data/audio/<post_id>.wav and record it in the entry.

    Resilient: a failure (bad text, model error) logs a warning and leaves audio_path null
    rather than aborting the run, so one bad story never blocks the rest.
    """
    tts_opts = settings.tts_opts
    lang_code = tts_opts.get("default_lang", "a")
    voices = tts_opts["voices"][lang_code]
    voice = pick_voice(voices, index)
    out_path = settings.audio_dir / f"{story.post_id}.wav"
    try:
        meta = synth.synthesize(
            entry.text["cleaned_body"],
            out_path,
            voice=voice,
            lang_code=lang_code,
            speed=float(tts_opts.get("speed", 1.0)),
        )
        entry.assets["audio_path"] = meta["audio_path"]
        entry.assets["audio"] = {k: meta[k] for k in ("voice", "lang_code", "duration_sec", "sample_rate")}
        entry.status = "tts_done"
    except Exception as exc:  # noqa: BLE001 -- keep the pipeline going on any TTS failure
        print(f"[brainrotbot] WARNING: TTS failed for {story.post_id} ({voice}/{lang_code}): {exc}")


def _add_background_video(entry: LedgerEntry, story: Story, maker: BackgroundVideoMaker, settings, index: int) -> None:
    """Build a vertical gameplay clip sized to the narration and record it in the entry.

    Trims to the real narrated duration (assets.audio.duration_sec); if TTS was skipped/failed,
    falls back to the word-count estimate so the step still works standalone. Resilient like
    `_add_audio`: any download/ffmpeg failure logs a warning and leaves background_video null.
    """
    audio = entry.assets.get("audio")
    duration = audio["duration_sec"] if audio else entry.text["est_speech_seconds"]
    sources = [s for s in (settings.video_opts.get("sources") or []) if s]  # drop blank entries
    if not duration or duration <= 0 or not sources:
        print(f"[brainrotbot] WARNING: skipping background video for {story.post_id} "
              f"(duration={duration}, sources={len(sources)}).")
        return
    source_url = pick_source(sources, index)
    out_path = settings.video_dir / f"{story.post_id}.mp4"
    try:
        meta = maker.make(source_url, float(duration), out_path)
        entry.assets["background_video"] = meta["path"]
        entry.assets["background"] = {
            k: meta[k] for k in
            ("source_url", "source_id", "start_sec", "duration_sec", "looped", "width", "height", "fps")
        }
        entry.status = "video_done"
    except Exception as exc:  # noqa: BLE001 -- one bad source/clip must not abort the run
        print(f"[brainrotbot] WARNING: background video failed for {story.post_id}: {exc}")


def _add_music_bed(entry: LedgerEntry, story: Story, catalogue: list[TrackMeta], settings) -> None:
    """Pick a random NCS instrumental track for this story and stash it in the entry.

    The actual mix happens later in `_add_final_video` (compose() reads `music_path`). Per-story
    randomness is intentional (user spec): every video gets a fresh track, not a deterministic
    rotation. A failed download is swallowed so the rest of the pipeline still produces a final
    video -- just without the music bed.
    """
    track = pick_track(catalogue)
    try:
        mp3 = download_track(track, settings.music_cache_dir)
        edit_opts = settings.edit_opts
        entry.assets["music_path"] = str(mp3)
        entry.assets["music"] = {
            "track_id": track.track_id,
            "title": track.title,
            "artist": track.artist,
            "genre": track.genre,
            "moods": track.moods,
            "page_url": track.page_url,
            "instrumental_url": track.instrumental_url,
            "volume_db": edit_opts.get("music_volume_db", -15.0),
            "ducked": bool(edit_opts.get("music_duck", True)),
        }
    except Exception as exc:  # noqa: BLE001 -- a flaky download must not kill the story
        print(f"[brainrotbot] WARNING: music download failed for {story.post_id} "
              f"({track.title} / {track.artist}): {exc}")


def _add_final_video(entry: LedgerEntry, story: Story, editor: VideoEditor, settings) -> None:
    """Mux the narration onto the background clip (+ outro, + music bed) into data/final/<post_id>.mp4.

    Needs both prior assets; if either is missing (TTS or background skipped/failed) it warns and
    leaves final_video null. Music is optional -- if `music_path` is set we hand it to compose()
    for the soft-bed mix, else compose() runs its original Step 4 paths. Resilient like the other
    steps: any ffmpeg failure is swallowed so one bad story never blocks the rest.
    """
    background = entry.assets.get("background_video")
    audio = entry.assets.get("audio_path")
    if not background or not audio:
        print(f"[brainrotbot] WARNING: skipping final video for {story.post_id} "
              f"(background={bool(background)}, audio={bool(audio)}).")
        return
    music_path = entry.assets.get("music_path")
    out_path = settings.final_dir / f"{story.post_id}.mp4"
    try:
        meta = editor.compose(
            Path(background), Path(audio), out_path,
            music_path=Path(music_path) if music_path else None,
        )
        entry.assets["final_video"] = meta["path"]
        entry.assets["edit"] = {
            k: meta[k] for k in
            ("has_outro", "has_music", "outro_file", "duration_sec", "width", "height", "fps")
        }
        entry.status = "edit_done"
    except Exception as exc:  # noqa: BLE001 -- one bad edit must not abort the run
        print(f"[brainrotbot] WARNING: final video failed for {story.post_id}: {exc}")


def _write_story_file(settings, story: Story, entry: LedgerEntry) -> None:
    path = settings.stories_dir / f"{story.post_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)


def _print_summary(entries: list[LedgerEntry]) -> None:
    if not entries:
        print("[brainrotbot] No new stories produced.")
        return
    print("\n=== Selected stories ===")
    for e in entries:
        t = e.text
        audio = e.assets.get("audio")
        # Show the real narrated duration/voice when TTS ran, else the word-count estimate.
        narration = (
            f"audio={audio['duration_sec']}s voice={audio['voice']}"
            if audio else f"~{t['est_speech_seconds']}s (no audio)"
        )
        bg = e.assets.get("background")
        video = f" bg={bg['source_id']}@{bg['start_sec']}s" if bg else ""
        edit = e.assets.get("edit")
        final = ""
        if edit:
            final = f" final={edit['duration_sec']}s"
            if edit.get("has_outro"):
                final += "+outro"
            if edit.get("has_music"):
                final += "+music"
        music = e.assets.get("music")
        music_tag = f" track={music['genre']}/{music['title']}" if music else ""
        print(
            f"- [{e.source['subreddit']}] feed_rank={e.source['feed_rank']} "
            f"words={t['word_count']} {narration}{video}{final}{music_tag} "
            f"replaced={len(t['banned_words_replaced'])}\n"
            f"    {t['title'][:90]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainrotbot", description="Reddit retrieval + TTS.")
    parser.add_argument("--settings", default=None, help="Path to settings.toml")
    parser.add_argument("--top-k", type=int, default=None, help="Override stories kept per run")
    parser.add_argument("--skip-tts", action="store_true", help="Skip text-to-speech (Step 1 only)")
    parser.add_argument("--skip-video", action="store_true", help="Skip background-video retrieval (Step 3)")
    parser.add_argument("--skip-edit", action="store_true", help="Skip editing/final-video assembly (Step 4)")
    parser.add_argument("--skip-music", action="store_true", help="Skip background-music bed (Step 5)")
    args = parser.parse_args(argv)
    run(settings_path=args.settings, top_k=args.top_k, skip_tts=args.skip_tts,
        skip_video=args.skip_video, skip_edit=args.skip_edit, skip_music=args.skip_music)
    return 0


if __name__ == "__main__":
    sys.exit(main())
