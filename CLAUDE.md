This codebase will be focused on automating TikTok content creation and uploading -- called brainrotbot. Specifically, the goal for the first version is to retrieve interesting stories from Reddit, then to convert this text to speech, to add a background video, and finally to edit the video and to upload it. This is a personal, single-user project (not meant to be cloned/installed by others), and it must run standalone: one command (`run.bat`) turns the bot on and runs the full pipeline end to end. In more detail:

## Status (what's built)

- **Step 1 — DONE**: Reddit retrieval (RSS) + text cleaning + analytics ledger. Output: per-story JSON in `data/stories/` + append-only `data/ledger.jsonl`. `LedgerEntry` already reserves `assets`/`upload`/`metrics`/`content_analysis` fields for later steps.
- **Step 2 — DONE**: Text-to-speech via **Kokoro-82M** (`src/brainrotbot/tts/`). Narrates each story's cleaned text to `data/audio/<post_id>.wav` (24 kHz), rotating a per-language voice pool (A/B testing) and recording voice/lang/duration in `assets.audio` + `assets.audio_path`; sets `status="tts_done"`. Config in `[tts]` (settings.toml). Requires the `tts` extra: `pip install -e ".[tts]"` (kokoro, soundfile, numpy, misaki[ja]/[zh] — the last two are pre-installed for planned JP/ZH post translation). **Runs in the `brainrotbot312` conda env (Python 3.12): Kokoro requires Python <3.13, so `run.bat` points there, not the original 3.13 `brainrotbot` env.** misaki auto-fetches its `en_core_web_sm` G2P model on first run (no manual espeak-ng install was needed). Use `--skip-tts` to run Step 1 alone. Verified end-to-end (real Reddit→WAV, voice rotation confirmed).
- **Step 3 — DONE**: background video (`src/brainrotbot/video/`). Per story, picks a source from the curated `[video].sources` pool (round-robin → A/B), downloads it once with **yt-dlp** to `data/video_cache/` (keyed by URL hash, reused across runs), trims a window the length of the narration (`assets.audio.duration_sec`; falls back to the word-count estimate if TTS was skipped) from a random offset in the source (no earlier than `intro_skip_sec`, default 5s, to skip the source's intro/title card), then center-crops 16:9→9:16 to `data/video/<post_id>.mp4` (silent; narration is muxed in Step 4). Records source/window in `assets.background` + `assets.background_video`; sets `status="video_done"`. Config in `[video]` (settings.toml). Requires the `video` extra: `pip install -e ".[video]"` (yt-dlp, imageio-ffmpeg). **ffmpeg is the static binary bundled by `imageio-ffmpeg`, not conda-forge's** — the conda-forge Windows ffmpeg build crashes with a DLL entrypoint error (`0xC0000139`), so it was removed; duration is read from `ffmpeg -i` (no ffprobe). **YouTube now gates downloads behind a "confirm you're not a bot" check; a JS runtime (Deno, conda-forge) is installed and required, and logged-in cookies are needed too** — set `[video].cookies_file` (a Netscape cookies.txt, most reliable on Windows) or `cookies_from_browser` (browser must be closed; modern Chromium app-bound encryption / a running browser locking its DB makes this flaky). Use `--skip-video` to skip. Trim/crop/intro-skip verified offline on a synthetic source; the real yt-dlp download path worked earlier this session but is currently bot-gated pending valid cookies.
- **Steps 4-6 — NOT STARTED**: editing, upload, analysis. The numbered sections below are the spec/intent, not status. Each step slots into `pipeline.run()`; `run.bat` stays the stable single entry point.

## Working practices

- **Commit to git frequently.** Commit after every small, independently useful change (a passing module, its test, a working sub-step) rather than batching everything into one commit at the end of a task. Small, self-contained commits keep history reviewable and make rollback cheap. Each commit message should describe the one useful thing it achieves.
- **Be concise.** When writing code, it is important to implement it concisely and efficiently. I.e., implement the minimum changes necessary. This helps in understanding the changes and reviewing version control. Also, write plentiful information-dense comments.

## Pipeline spec

1. **Story retrieval from Reddit**

On Reddit, many fascinating, provocative, despicable, or otherwise interesting stories circulate, not only in r/stories, but literally on any sub. The goal here is to find such stories, and retrieve the text. This is done via Reddit's public **RSS feeds** (e.g. `/r/<sub>/top/.rss?t=week`), which need no auth and no app registration and carry the full self-text. (The anonymous `.json` endpoint is now 403-blocked; the Data API is gated/legacy and steered toward moderation use cases; and the Developer Platform / Devvit at developers.reddit.com is for apps hosted *on* Reddit, not external bots -- so none of those are used.) RSS omits upvote/comment counts, so selection uses the feed's popularity ordering. Selection could later be refined using Gemma4-E2B. Use judgement here.

Subsequently, this text should be filtered on provocative words TikTok does not allow, which should be replaced by similar words. This can be automated using either simple regex or Gemma4-E2B. Use judgement here.

2. **Text-to-speech conversion**

Once the text is retrieved and cleaned, it should be converted to speech, so that TikTok users can listen to the content. Preferably this should be automated using a small local TTS model (e.g., K2-fsa OmniVoice).

3. **Background video retrieval**

Good brainrot needs some attention-retaining background videos, e.g., Minecraft/Fortnite/Subway-Surfers gameplay. Automate downloading these videos from e.g., YouTube, again using, e.g., Playwright and perhaps a yt2mp4 service. The range of video content suitable for this project is large, so our brainrotbot may be creative in selecting the video.

Important here is that the video is trimmed properly, i.e., that only the "important" content of the video is visible. To perform this edit, perhaps we should use a VLM such as Gemma4-E2B.

4. **Editing**

Once the three parts are retrieved, the audio should be overlayed on the video. Also, add a standard outro shot showing a call to action to follow our page. If the video is significantly long, add "Follow for part 2/...". Notice, however, that I prefer all videos to be at least one minute, although both shorter and longer videos are tolerated.

5. **Uploading**

Now that the video is prepared, upload it to TikTok. I will later provide the account information, including login. This should again be automated using e.g., Playwright. Note that TikTok already provides a tool for adding subtitles, please turn this on.

6. **Analysis**

When the bot is up and running, it should automatically analyze the success of uploaded content. This includes A/B testing or something similar. The goal is to find what works. Therefore, for each uploaded content, an entry should be added in our ledger, describing its content analytically, e.g., the style of the text, the broad topic, the background video, etc. 