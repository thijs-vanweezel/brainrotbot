This codebase will be focused on automating TikTok content creation and uploading -- called brainrotbot. Specifically, the goal for the first version is to retrieve interesting stories from Reddit, then to convert this text to speech, to add a background video, and finally to edit the video and to upload it. This is a personal, single-user project (not meant to be cloned/installed by others), and it must run standalone: one command (`run.bat`) turns the bot on and runs the full pipeline end to end. In more detail:

## Status (what's built)

- **Step 1 — DONE**: Reddit retrieval (RSS) + text cleaning + analytics ledger. Output: per-story JSON in `data/stories/` + append-only `data/ledger.jsonl`. `LedgerEntry` already reserves `assets`/`upload`/`metrics`/`content_analysis` fields for later steps.
- **Steps 2-6 — NOT STARTED**: TTS, background video, editing, upload, analysis. The numbered sections below are the spec/intent, not status. Each step slots into `pipeline.run()`; `run.bat` stays the stable single entry point.

## Working practices

- **Commit to git frequently.** Commit after every small, independently-useful change (a passing module + its test, a working sub-step) rather than batching everything into one commit at the end of a task. Small, self-contained commits keep history reviewable and make rollback cheap. Each commit message should describe the one useful thing it achieves.

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

Once the three parts are retrieved, the audio should be overlayed on the video. Also, add a standard outro shot showing a call to action to follow our page.

5. **Uploading**

Now that the video is prepared, upload it to TikTok. I will later provide the account information, including login. This should again be automated using e.g., Playwright. Note that TikTok already provides a tool for adding subtitles, please turn this on.

6. **Analysis**

When the bot is up and running, it should automatically analyze the success of uploaded content. This includes A/B testing or something similar. The goal is to find what works. Therefore, for each uploaded content, an entry should be added in our ledger, describing its content analytically, e.g., the style of the text, the broad topic, the background video, etc. 