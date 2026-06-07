"""Step 7: upload finished videos to TikTok by driving the web Studio uploader (Playwright).

We use browser automation, NOT TikTok's official Content Posting API, for three reasons:
  - the API has **no parameter to enable auto-captions/subtitles** (a UI-only toggle the user wants ON);
  - unaudited API clients are forced to SELF_ONLY (private) posting -- public posting needs TikTok's
    app audit, unsuitable for a personal single-user bot;
  - the API returns only a publish_id, not the public video URL we must write back to the ledger.

A persistent Chromium profile (config/tiktok_profile/, gitignored) keeps the logged-in session and the
full browser fingerprint across runs -- the most robust setup against TikTok's bot checks. Log in once
via `run.bat --tiktok-login`; later drains reuse that profile.

The Studio DOM is not a stable API: the selectors below are best-effort and may need re-pinning when
TikTok ships UI changes. They're isolated as named constants for quick fixups, and every fragile step is
wrapped so a single UI hiccup degrades gracefully (e.g. cover falls back to TikTok's auto-frame) rather
than aborting the whole batch. `playwright` is imported lazily so the package imports without the extra.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload"
CONTENT_URL = "https://www.tiktok.com/tiktokstudio/content"
LOGIN_URL = "https://www.tiktok.com/login"

# --- Best-effort Studio selectors (subject to change -- see module docstring) --------------------
_FILE_INPUT = "input[type=file]"                       # hidden file picker on the upload page
_CAPTION_EDITOR = "div[contenteditable='true']"        # DraftJS caption box
_POST_BUTTON = "button:has-text('Post')"               # final submit
_REPLACE_BUTTON = "button:has-text('Replace')"         # appears once a video finished processing
_VIDEO_HREF = "a[href*='/video/']"                     # links that carry a posted video id
_VIDEO_ID_RE = re.compile(r"/video/(\d+)")


class TikTokUploader:
    """Drives the TikTok Studio web uploader. Use as a context manager (one browser per drain batch).

        with TikTokUploader(session_dir=..., ...) as up:
            meta = up.upload(video, cover, caption)
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        upload_url: str = UPLOAD_URL,
        headless: bool = False,
        privacy: str = "public",
        subtitles: bool = True,
        set_cover: bool = True,
        nav_timeout_sec: float = 120.0,
    ):
        self.session_dir = Path(session_dir)
        self.upload_url = upload_url
        self.headless = headless
        self.privacy = privacy
        self.subtitles = subtitles
        self.set_cover = set_cover
        self.nav_timeout_ms = int(nav_timeout_sec * 1000)
        self._pw = None
        self._ctx = None  # Playwright persistent BrowserContext

    # --- lifecycle -------------------------------------------------------------------------------
    def _launch(self, *, headless: bool):
        """Start Playwright + a persistent Chromium context bound to our profile dir."""
        from playwright.sync_api import sync_playwright  # lazy: keeps the extra optional

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self._ctx = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.session_dir),
            headless=headless,
            viewport={"width": 1280, "height": 900},
        )
        self._ctx.set_default_timeout(self.nav_timeout_ms)

    def _close(self):
        try:
            if self._ctx is not None:
                self._ctx.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
            self._ctx = self._pw = None

    def __enter__(self) -> "TikTokUploader":
        self._launch(headless=self.headless)
        return self

    def __exit__(self, *exc):
        self._close()

    # --- one-time interactive login --------------------------------------------------------------
    def login(self) -> None:
        """Open a headed browser at TikTok's login page and block until the user finishes.

        Run interactively (`run.bat --tiktok-login`): log in (and pass any captcha) in the window that
        opens, then press Enter in the terminal. The session persists in the profile dir for later runs.
        """
        self._launch(headless=False)
        try:
            page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            page.goto(LOGIN_URL)
            print("[brainrotbot] A browser opened on TikTok's login page.")
            print("[brainrotbot] Log in there (handle any captcha), then return here and press Enter.")
            input("[brainrotbot] Press Enter once you are logged in to save the session... ")
        finally:
            self._close()
        print(f"[brainrotbot] TikTok session saved to {self.session_dir}")

    # --- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _pause(lo: float = 0.6, hi: float = 1.6) -> None:
        """Small randomized human-like delay between UI actions."""
        time.sleep(random.uniform(lo, hi))

    def _ensure_logged_in(self, page) -> None:
        """Heuristic: the upload page bounces to /login when there's no valid session."""
        if "/login" in page.url:
            raise RuntimeError("not logged in to TikTok -- run `run.bat --tiktok-login` first")

    def _enable_captions(self, page) -> bool:
        """Best-effort: turn on TikTok's auto-generated captions toggle. Returns True if flipped.

        The control is a Switch near a 'Captions' label in the Studio side panel; its exact markup
        shifts between Studio versions, so we try a couple of strategies and never hard-fail.
        """
        try:
            # Some Studio builds hide captions behind a "Show more" / "More options" expander.
            for label in ("Show more", "More options"):
                more = page.get_by_text(label, exact=False)
                if more.count() and more.first.is_visible():
                    more.first.click()
                    self._pause()
                    break
            switches = page.get_by_role("switch")
            for i in range(switches.count()):
                sw = switches.nth(i)
                name = (sw.get_attribute("aria-label") or "").lower()
                # Fall back to the nearest text if the switch itself is unlabeled.
                if "caption" not in name:
                    try:
                        name = sw.locator("xpath=ancestor::*[self::label or self::div][1]").inner_text().lower()
                    except Exception:  # noqa: BLE001
                        name = name
                if "caption" in name:
                    if (sw.get_attribute("aria-checked") or "").lower() != "true":
                        sw.click()
                        self._pause()
                    return True
        except Exception as exc:  # noqa: BLE001 -- captions are a nice-to-have, never abort over them
            print(f"[brainrotbot]   (captions toggle not flipped: {exc})")
        return False

    def _set_cover(self, page, cover_path: Path) -> bool:
        """Best-effort custom cover from the Step 6 PNG. Returns True on success."""
        try:
            for label in ("Edit cover", "Select cover", "Cover"):
                btn = page.get_by_text(label, exact=False)
                if btn.count() and btn.first.is_visible():
                    btn.first.click()
                    self._pause()
                    break
            else:
                return False
            up_tab = page.get_by_text(re.compile(r"upload\s+cover", re.I))
            if up_tab.count():
                up_tab.first.click()
                self._pause()
            # The cover dialog has its own file input; pick the last one on the page to avoid the
            # main video input.
            inputs = page.locator(_FILE_INPUT)
            inputs.nth(inputs.count() - 1).set_input_files(str(cover_path))
            self._pause(1.0, 2.0)
            confirm = page.get_by_role("button", name=re.compile(r"confirm|done|save|apply", re.I))
            if confirm.count():
                confirm.first.click()
                self._pause()
            return True
        except Exception as exc:  # noqa: BLE001 -- fall back to TikTok's auto frame
            print(f"[brainrotbot]   (custom cover not set, using auto frame: {exc})")
            return False

    def _set_public(self, page) -> None:
        """Best-effort: ensure visibility is 'Everyone' (Studio usually defaults to public already)."""
        if self.privacy != "public":
            return
        try:
            opt = page.get_by_text("Everyone", exact=False)
            if opt.count() and opt.first.is_visible():
                opt.first.click()
                self._pause()
        except Exception:  # noqa: BLE001
            pass

    def _capture_url(self, page) -> tuple[str | None, str | None]:
        """After posting, resolve the live video URL+id. Best-effort; returns (url, id) or (None, None)."""
        # 1) A "View" link sometimes appears in the success toast/modal.
        try:
            link = page.locator(_VIDEO_HREF)
            link.first.wait_for(timeout=8000)
            href = link.first.get_attribute("href")
            if href:
                return self._normalize(href)
        except Exception:  # noqa: BLE001
            pass
        # 2) Fall back to the newest item in the Studio content list.
        try:
            page.goto(CONTENT_URL)
            link = page.locator(_VIDEO_HREF)
            link.first.wait_for(timeout=15000)
            href = link.first.get_attribute("href")
            if href:
                return self._normalize(href)
        except Exception:  # noqa: BLE001
            pass
        return None, None

    @staticmethod
    def _normalize(href: str) -> tuple[str | None, str | None]:
        url = href if href.startswith("http") else f"https://www.tiktok.com{href}"
        m = _VIDEO_ID_RE.search(href)
        return url, (m.group(1) if m else None)

    # --- main entry ------------------------------------------------------------------------------
    def upload(self, video_path: Path, cover_path: Path | None, caption: str) -> dict:
        """Post one video; return {url, tiktok_id, posted_at, public, captions_on, cover_set}.

        Raises on hard failures (no session, processing never completed, Post never succeeded) so the
        queue marks the video failed and retries it on the next drain. Cosmetic steps (captions, cover,
        visibility) degrade quietly.
        """
        if self._ctx is None:
            raise RuntimeError("TikTokUploader used outside its context manager (`with uploader:`)")
        page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        page.goto(self.upload_url)
        self._ensure_logged_in(page)

        # 1) Hand the file to the hidden input and wait for TikTok to finish processing it.
        page.locator(_FILE_INPUT).first.set_input_files(str(video_path))
        page.locator(_REPLACE_BUTTON).first.wait_for(timeout=self.nav_timeout_ms)  # "Replace" => done
        self._pause(1.0, 2.0)

        # 2) Caption = title + hashtags. Clear the placeholder text, then type.
        editor = page.locator(_CAPTION_EDITOR).first
        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Delete")
        editor.type(caption, delay=15)
        self._pause()

        # 3) Subtitles (requirement #3), 4) custom cover, 5) public visibility -- all best-effort.
        captions_on = self._enable_captions(page) if self.subtitles else False
        cover_set = bool(cover_path) and self.set_cover and self._set_cover(page, Path(cover_path))
        self._set_public(page)

        # 6) Post and confirm it went through.
        post = page.locator(_POST_BUTTON).first
        post.wait_for(timeout=self.nav_timeout_ms)
        post.click()
        # Success = the post button goes away / a confirmation toast shows. Give it room.
        try:
            page.get_by_text(re.compile(r"your video|posted|uploaded|manage", re.I)).first.wait_for(
                timeout=self.nav_timeout_ms
            )
        except Exception:  # noqa: BLE001 -- some builds just navigate; URL capture still tries
            pass
        self._pause(1.5, 2.5)

        url, tiktok_id = self._capture_url(page)
        return {
            "url": url,
            "tiktok_id": tiktok_id,
            "posted_at": time.time(),
            "public": self.privacy == "public",
            "captions_on": captions_on,
            "cover_set": cover_set,
        }
