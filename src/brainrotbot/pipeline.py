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


def run(settings_path=None, top_k: int | None = None) -> list[LedgerEntry]:
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

    settings.stories_dir.mkdir(parents=True, exist_ok=True)
    entries: list[LedgerEntry] = []
    for story in selected:
        entry = _process_story(story, settings)
        _write_story_file(settings, story, entry)
        append_entry(settings.ledger_path, entry)
        entries.append(entry)

    _print_summary(entries)

    # Future steps consume `entries` here and fill the ledger's assets/upload/metrics:
    #   2. text-to-speech    -> entry.assets["audio_path"]
    #   3. background video  -> entry.assets["background_video"]
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
        print(
            f"- [{e.source['subreddit']}] feed_rank={e.source['feed_rank']} "
            f"words={t['word_count']} ~{t['est_speech_seconds']}s "
            f"replaced={len(t['banned_words_replaced'])}\n"
            f"    {t['title'][:90]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainrotbot", description="Step 1: Reddit retrieval.")
    parser.add_argument("--settings", default=None, help="Path to settings.toml")
    parser.add_argument("--top-k", type=int, default=None, help="Override stories kept per run")
    args = parser.parse_args(argv)
    run(settings_path=args.settings, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
