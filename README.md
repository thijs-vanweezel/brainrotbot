# brainrotbot

My personal bot for automating TikTok "brainrot" content. Goal (see `CLAUDE.md`):
Reddit story → clean text → text-to-speech → background video → edit → upload → analyze,
all driven by **one command**.

**Status: Steps 1-3 done** — Reddit story retrieval + text cleaning + the analytics ledger,
text-to-speech narration (Kokoro-82M) to `data/audio/<post_id>.wav`, and a background gameplay
clip (`data/video/<post_id>.mp4`) trimmed to the narration and cropped to vertical 9:16.

## Run the bot

```
run.bat
```

Double-click `run.bat`, or run it from a terminal (extra args pass through, e.g.
`run.bat --top-k 3`, `run.bat --skip-tts` for Step 1 only, or `run.bat --skip-video` to skip
Step 3). It executes the full pipeline in the `brainrotbot312` conda env.

Equivalent direct invocation:

```powershell
conda activate brainrotbot312
python -m brainrotbot.pipeline
```

Output:
- `data/stories/<post_id>.json` — full record per selected story (raw + cleaned text).
- `data/audio/<post_id>.wav` — narrated audio (24 kHz), with a rotating voice per story.
- `data/video/<post_id>.mp4` — silent 9:16 background clip, trimmed to the narration length from
  a random offset in the source gameplay and center-cropped to vertical.
- `data/video_cache/<hash>.<ext>` — source gameplay videos, downloaded once and reused.
- `data/ledger.jsonl` — append-only ledger, one line per story. TTS fills `assets.audio_path`
  / `assets.audio` (`status="tts_done"`); Step 3 fills `assets.background_video` / `assets.background`
  (`status="video_done"`); later steps (upload, analytics) fill the remaining reserved fields.

## One-time setup

Runs in the `brainrotbot312` conda env (Python 3.12 — Kokoro TTS requires Python <3.13), with
the TTS and video extras installed: `pip install -e ".[tts,video]"`. GPU is optional (the 82M
model is fast on CPU); a CUDA build of torch is installed for the planned later steps.

Step 3 needs no system ffmpeg: the `video` extra pulls in `yt-dlp` (clip sourcing) and
`imageio-ffmpeg`, which bundles a static ffmpeg binary (avoids the conda-forge Windows DLL
crash). Source gameplay videos are listed in `[video].sources` (`config/settings.toml`).

YouTube downloads need two extra things now: a **JS runtime** (Deno, `conda install -c
conda-forge deno`) and **logged-in cookies** to clear the "confirm you're not a bot" gate.
Set `[video].cookies_file` to an exported `cookies.txt` (most reliable on Windows — use a
browser extension like "Get cookies.txt LOCALLY"), or `cookies_from_browser = "edge"`/`"chrome"`
with that browser fully closed (a running browser locks its cookie DB).

**No Reddit credentials are needed** — stories are pulled from public RSS feeds
(`/r/<sub>/top/.rss`). This sidesteps Reddit's API entirely: the anonymous `.json` endpoint
is now 403-blocked, and the Data API is gated/legacy (steered toward moderation use cases via
the on-Reddit Developer Platform / Devvit, which doesn't fit a standalone external bot).

The trade-off: RSS exposes the full story text but not upvote/comment counts, so selection
uses the feed's own popularity ordering (`feed_rank`) and those numeric fields stay null in
the ledger.

## Planned

- **Background music** — the editing step (Step 4) should mix a low-volume music bed under the
  narration (ducked so the voice stays clear). Tracks would come from a curated, copyright-safe
  pool (rotated per video for A/B testing, like the voices and background sources), with the
  chosen track logged in the ledger for Step 6 analytics. Not yet implemented.

## Configuration

- `config/settings.toml` — subreddits, time window, score/length thresholds, paths.
- `resources/banned_words.toml` — TikTok-sensitive word → euphemism map (editable data).

## Tests

```powershell
pytest
```

Text cleaning, banned-word filtering, and story selection are covered offline (no network).
