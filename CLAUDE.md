This codebase will be focused on automating TikTok content creation and uploading -- called brainrotbot. Specifically, the goal for the first version is to retrieve interesting stories from Reddit, then to convert this text to speech, to add a background video, and finally to edit the video and to upload it. In more detail:

1. **Story retrieval from Reddit**

On Reddit, many fascinating, provocative, despicable, or otherwise interesting stories circulate, not only in r/stories, but literally on any sub. The goal here is to find such stories, and retrieve the text. This process can be automated using e.g., Playwright. Potential selection can occur based on either post popularity or using Gemma4-E2B. Use judgement here.

Subsequently, this text should be filtered on provocative words TikTok does not allow, which should be replaced by similar words. This can be automated using either simple regex or Gemma4-E2B. Use judgement here.

2. **Text-to-speech conversion**

Once the text is retrieved and cleaned, it should be converted to speech, so that TikTok users can listen to the content. Preferably this should be automated using a small local TTS model (e.g., K2-fsa OmniVoice).

3. **Background video retrieval**

Good brainrot needs some attention-retaining background videos, e.g., Minecraft/Fortnite/Subway-Surfers gameplay. Automate downloading these videos from e.g., YouTube, again using, e.g., Playwright and perhaps a yt2mp4 service. The range of video content suitable for this project is large, so our brainrotbot may be creative in selecting the video.

Important here is that the video is trimmed properly, i.e., that only the "important" content of the video is visible. To perform this edit, perhaps we should use a VLM such as Gemma4-E2B.

4. **Editing**

Once the three parts are retrieved, the audio should be overlayed on the video. Also, add a standard outro shot showing a call to action to follow our page.

5. **Uploading**

Now that the video is prepared, upload it to TikTok. I will later provide the account information, including login. This should again be automated using e.g., Playwright.

6. **Analysis**

When the bot is up and running, it should automatically analyze the success of uploaded content. This includes A/B testing or something similar. The goal is to find what works. Therefore, for each uploaded content, an entry should be added in our ledger, describing its content analytically, e.g., the style of the text, the broad topic, the background video, etc. 