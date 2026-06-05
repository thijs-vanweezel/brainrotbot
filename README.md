# brainrotbot

My personal bot for automating TikTok "brainrot" content. Goal (see `CLAUDE.md`):
Reddit story → clean text → text-to-speech → background video → edit → upload → analyze,
all driven by **one command**.

**Status: Step 1 done** — Reddit story retrieval + text cleaning + the analytics ledger.

## Run the bot

```
run.bat
```

Double-click `run.bat`, or run it from a terminal (extra args pass through, e.g.
`run.bat --top-k 3`). It executes the full pipeline in the `brainrotbot` conda env.

Equivalent direct invocation:

```powershell
conda activate brainrotbot
python -m brainrotbot.pipeline
```

Output:
- `data/stories/<post_id>.json` — full record per selected story (raw + cleaned text).
- `data/ledger.jsonl` — append-only ledger, one line per story. Later steps (TTS, video,
  upload, analytics) fill the reserved `assets` / `upload` / `metrics` fields.

## One-time setup

The `brainrotbot` conda env already exists with deps installed (`pip install -e .` done).
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
