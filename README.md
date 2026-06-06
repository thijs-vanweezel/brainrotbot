# brainrotbot

My personal bot for automating TikTok "brainrot" content. Goal (see `CLAUDE.md`):
Reddit story → clean text → text-to-speech → background video → edit → upload → analyze,
all driven by **one command**.

**Status: Steps 1-2 done** — Reddit story retrieval + text cleaning + the analytics ledger,
and text-to-speech narration (Kokoro-82M) to `data/audio/<post_id>.wav`.

## Run the bot

```
run.bat
```

Double-click `run.bat`, or run it from a terminal (extra args pass through, e.g.
`run.bat --top-k 3`, or `run.bat --skip-tts` for Step 1 only). It executes the full pipeline
in the `brainrotbot312` conda env.

Equivalent direct invocation:

```powershell
conda activate brainrotbot312
python -m brainrotbot.pipeline
```

Output:
- `data/stories/<post_id>.json` — full record per selected story (raw + cleaned text).
- `data/audio/<post_id>.wav` — narrated audio (24 kHz), with a rotating voice per story.
- `data/ledger.jsonl` — append-only ledger, one line per story. TTS fills `assets.audio_path`
  / `assets.audio` and sets `status="tts_done"`; later steps (video, upload, analytics) fill the
  remaining reserved `assets` / `upload` / `metrics` fields.

## One-time setup

Runs in the `brainrotbot312` conda env (Python 3.12 — Kokoro TTS requires Python <3.13), with
the TTS extra installed: `pip install -e ".[tts]"`. GPU is optional (the 82M model is fast on
CPU); a CUDA build of torch is installed for the planned later steps.

**No Reddit credentials are needed** — stories are pulled from public RSS feeds
(`/r/<sub>/top/.rss`). This sidesteps Reddit's API entirely: the anonymous `.json` endpoint
is now 403-blocked, and the Data API is gated/legacy (steered toward moderation use cases via
the on-Reddit Developer Platform / Devvit, which doesn't fit a standalone external bot).

The trade-off: RSS exposes the full story text but not upvote/comment counts, so selection
uses the feed's own popularity ordering (`feed_rank`) and those numeric fields stay null in
the ledger.

## Configuration

- `config/settings.toml` — subreddits, time window, score/length thresholds, paths.
- `resources/banned_words.toml` — TikTok-sensitive word → euphemism map (editable data).

## Tests

```powershell
pytest
```

Text cleaning, banned-word filtering, and story selection are covered offline (no network).
