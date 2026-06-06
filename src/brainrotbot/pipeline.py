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

from .config import load_settings
from .ledger import append_entry, existing_post_ids
from .models import LedgerEntry, Story
from .reddit.fetch import fetch_candidates
from .reddit.select import select_stories
from .text.clean import clean_text
from .text.filter_words import filter_banned_words
from .tts.synthesize import KokoroSynthesizer, pick_voice
from .video.background import BackgroundVideoMaker, pick_source


def run(
    settings_path=None,
    top_k: int | None = None,
    skip_tts: bool = False,
    skip_video: bool = False,
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

    settings.stories_dir.mkdir(parents=True, exist_ok=True)
    entries: list[LedgerEntry] = []
    for index, story in enumerate(selected):
        entry = _process_story(story, settings)
        if synth is not None:
            _add_audio(entry, story, synth, settings, index)
        if maker is not None:
            _add_background_video(entry, story, maker, settings, index)
        _write_story_file(settings, story, entry)
        append_entry(settings.ledger_path, entry)
        entries.append(entry)

    _print_summary(entries)

    # Future steps consume `entries` here and fill the ledger's assets/upload/metrics:
    #   2. text-to-speech    -> entry.assets["audio_path"]        (DONE)
    #   3. background video  -> entry.assets["background_video"]  (DONE)
    #   4. editing           -> entry.assets["final_video"]
    #   5. upload to TikTok   -> entry.upload[...]
    #   6. analysis           -> entry.metrics[...] + entry.content_analysis[...]
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
        print(
            f"- [{e.source['subreddit']}] feed_rank={e.source['feed_rank']} "
            f"words={t['word_count']} {narration}{video} "
            f"replaced={len(t['banned_words_replaced'])}\n"
            f"    {t['title'][:90]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainrotbot", description="Reddit retrieval + TTS.")
    parser.add_argument("--settings", default=None, help="Path to settings.toml")
    parser.add_argument("--top-k", type=int, default=None, help="Override stories kept per run")
    parser.add_argument("--skip-tts", action="store_true", help="Skip text-to-speech (Step 1 only)")
    parser.add_argument("--skip-video", action="store_true", help="Skip background-video retrieval (Step 3)")
    args = parser.parse_args(argv)
    run(settings_path=args.settings, top_k=args.top_k, skip_tts=args.skip_tts, skip_video=args.skip_video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
