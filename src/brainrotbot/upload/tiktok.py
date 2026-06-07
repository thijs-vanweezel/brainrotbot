"""Step 7: upload finished videos to TikTok by driving the web Studio uploader (Playwright).

We use browser automation, NOT TikTok's official Content Posting API, for three reasons:
  - the API has **no parameter to enable auto-captions/subtitles** (a UI-only toggle the user wants ON);
  - unaudited API clients are forced to SELF_ONLY (private) posting -- public posting needs TikTok's
    app audit, unsuitable for a personal single-user bot;
  - the API returns only a publish_id, not the public video URL we must write back to the ledger.

**Auth is by imported cookies, not automated login.** TikTok's login flow is bot-walled ("maximum
attempts reached") regardless of engine -- it blocks the automated browser *environment*, not the
password. So instead we inject a `cookies.txt` exported from the browser where the user is already logged
into TikTok (same "Get cookies.txt LOCALLY" workflow the project uses for YouTube), skipping login
entirely. Lightweight stealth tweaks (hide navigator.webdriver, drop the AutomationControlled flag) make
the upload/post actions look less automated too. A persistent profile (config/tiktok_profile/<engine>/,
gitignored) is still launched but is no longer the auth source; `run.bat --tiktok-login` survives as a
manual fallback, and `run.bat --tiktok-check` verifies the cookies without needing a finished video.

The Studio DOM is not a stable API: the selectors below are best-effort and may need re-pinning when
TikTok ships UI changes. They're isolated as named constants for quick fixups, and every fragile step is
wrapped so a single UI hiccup degrades gracefully (e.g. cover falls back to TikTok's auto-frame) rather
than aborting the whole batch. `playwright` is imported lazily so the package imports without the extra.
"""

from __future__ import annotations

import http.cookiejar
import random
import re
import time
from pathlib import Path

# Injected once per page so TikTok's bot checks don't see the automation flag.
_STEALTH_JS = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"

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
# A real "the post finished" signal -- only trust these, not the upload page's ambient nav text.
_POSTED_TEXT = re.compile(r"your video has been (uploaded|posted)|posted to|view profile", re.I)
# Buttons that may appear in a post-click confirmation/content-check modal (best-effort, click to proceed).
_CONFIRM_POST = re.compile(r"^(post now|post|continue|got it|confirm)$", re.I)


class TikTokUploader:
    """Drives the TikTok Studio web uploader. Use as a context manager (one browser per drain batch).

        with TikTokUploader(session_dir=..., ...) as up:
            meta = up.upload(video, cover, caption)
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        browser: str = "chromium",
        cookies_file: str = "",
        user_agent: str = "",
        upload_url: str = UPLOAD_URL,
        headless: bool = False,
        privacy: str = "public",
        subtitles: bool = True,
        set_cover: bool = True,
        nav_timeout_sec: float = 120.0,
        completion_timeout_sec: float = 300.0,
        debug: bool = False,
        debug_dir: Path | None = None,
    ):
        self.session_dir = Path(session_dir)
        self.browser = browser
        self.cookies_file = cookies_file
        self.user_agent = user_agent
        self.upload_url = upload_url
        self.headless = headless
        self.privacy = privacy
        self.subtitles = subtitles
        self.set_cover = set_cover
        self.nav_timeout_ms = int(nav_timeout_sec * 1000)
        # TikTok's post-click content check can take minutes; wait this long for a real "posted" signal.
        self.completion_timeout_sec = completion_timeout_sec
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self._pw = None
        self._ctx = None  # Playwright persistent BrowserContext

    # --- lifecycle -------------------------------------------------------------------------------
    def _launch(self, *, headless: bool):
        """Start Playwright + a persistent context (chromium/firefox/webkit), inject cookies + stealth.

        The profile is namespaced per engine (session_dir/<browser>/) because the on-disk user-data
        formats are incompatible. Auth comes from the imported cookies (see _load_cookies); the profile
        is just a stable browser to attach them to. Stealth: drop Chromium's AutomationControlled flag
        and mask navigator.webdriver so TikTok's bot checks don't flag the upload session.
        """
        from playwright.sync_api import sync_playwright  # lazy: keeps the extra optional

        profile_dir = self.session_dir / self.browser
        profile_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        engine = getattr(self._pw, self.browser)  # chromium | firefox | webkit
        kwargs = dict(
            user_data_dir=str(profile_dir),
            headless=headless,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        if self.user_agent:
            kwargs["user_agent"] = self.user_agent
        if self.browser == "chromium":  # Chromium-only flag (Firefox/WebKit reject unknown args)
            kwargs["args"] = ["--disable-blink-features=AutomationControlled"]
        self._ctx = engine.launch_persistent_context(**kwargs)
        self._ctx.set_default_timeout(self.nav_timeout_ms)
        self._ctx.add_init_script(_STEALTH_JS)
        cookies = self._load_cookies()
        if cookies:
            self._ctx.add_cookies(cookies)
            print(f"[brainrotbot] Injected {len(cookies)} TikTok cookies from {self.cookies_file}")

    def _load_cookies(self) -> list[dict]:
        """Parse the Netscape cookies.txt into Playwright add_cookies dicts (empty if unset/missing).

        Uses stdlib MozillaCookieJar; falls back to a manual tab-split parse when the file lacks the
        Netscape magic header (some exporters omit it). httpOnly/sameSite are not emitted -- Playwright
        still *sends* the cookies, which is all the session needs.
        """
        if not self.cookies_file or not Path(self.cookies_file).is_file():
            return []
        out: list[dict] = []
        jar = http.cookiejar.MozillaCookieJar()
        try:
            jar.load(self.cookies_file, ignore_discard=True, ignore_expires=True)
            for c in jar:
                out.append({
                    "name": c.name, "value": c.value,
                    "domain": c.domain, "path": c.path or "/",
                    "secure": bool(c.secure),
                    "expires": float(c.expires) if c.expires else -1,
                })
        except (http.cookiejar.LoadError, OSError):
            for line in Path(self.cookies_file).read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                f = line.split("\t")
                if len(f) >= 7:  # domain, flag, path, secure, expires, name, value
                    out.append({
                        "name": f[5], "value": f[6].strip(),
                        "domain": f[0], "path": f[2] or "/",
                        "secure": f[3].upper() == "TRUE",
                        "expires": float(f[4]) if f[4].lstrip("-").isdigit() else -1,
                    })
        return out

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

    def check(self) -> bool:
        """Open the upload page with the imported cookies and report whether the session is valid.

        Exposed via `run.bat --tiktok-check` -- verifies cookies without needing a finished video.
        Returns True when the Studio loads logged-in (no bounce to /login).
        """
        self._launch(headless=self.headless)
        try:
            page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            page.goto(self.upload_url)
            self._pause(2.0, 3.0)
            ok = "/login" not in page.url
            if ok:
                print("[brainrotbot] TikTok session OK -- Studio loaded logged in.")
            else:
                print("[brainrotbot] NOT logged in -- export a fresh tiktok.com cookies.txt "
                      f"to {self.cookies_file or '[upload].cookies_file'} and retry.")
            return ok
        finally:
            self._close()

    # --- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _pause(lo: float = 0.6, hi: float = 1.6) -> None:
        """Small randomized human-like delay between UI actions."""
        time.sleep(random.uniform(lo, hi))

    def _dump(self, page, label: str) -> None:
        """Debug only: save a screenshot + HTML + URL at a milestone, to pin selectors from the real DOM."""
        if not self.debug or self.debug_dir is None:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{int(time.time())}_{label}"
            page.screenshot(path=str(self.debug_dir / f"{stamp}.png"), full_page=True)
            (self.debug_dir / f"{stamp}.html").write_text(page.content(), encoding="utf-8")
            (self.debug_dir / f"{stamp}.url.txt").write_text(page.url, encoding="utf-8")
            print(f"[brainrotbot]   [debug] dumped {stamp} ({page.url})")
        except Exception as exc:  # noqa: BLE001 -- debugging must never break the upload
            print(f"[brainrotbot]   [debug] dump failed at {label}: {exc}")

    def _ensure_logged_in(self, page) -> None:
        """Heuristic: the upload page bounces to /login when there's no valid session."""
        if "/login" in page.url:
            raise RuntimeError(
                "not logged in to TikTok -- export a fresh tiktok.com cookies.txt to "
                f"{self.cookies_file or '[upload].cookies_file'} (verify with `run.bat --tiktok-check`)"
            )

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

    def _confirm_post(self, page) -> None:
        """Best-effort: clear a post-click confirmation/content-check modal so the post proceeds.

        TikTok sometimes shows a small modal after Post (e.g. content-check / "Post now"). Click its
        confirm button if one is visible; never hard-fail (the account already enabled content checks).
        """
        try:
            self._pause(1.0, 2.0)
            btn = page.get_by_role("button", name=_CONFIRM_POST)
            for i in range(btn.count()):
                if btn.nth(i).is_visible():
                    btn.nth(i).click()
                    self._pause()
                    return
        except Exception:  # noqa: BLE001
            pass

    def _await_completion(self, page) -> tuple[str | None, str | None]:
        """Block until TikTok finishes the content check and the post lands; return (url, id).

        This is the fix for the original bug: we must NOT navigate away or close while the check runs.
        Poll (up to completion_timeout_sec) for a real "posted" signal -- a redirect to the content
        manager, a success toast, or a /video/<id> link -- then read the URL. Falls back to opening the
        content list (now safe, post is done) and grabbing the newest video link. (None, None) if it
        never confirms -- the caller then treats the upload as unconfirmed and keeps the media.
        """
        deadline = time.time() + self.completion_timeout_sec
        while time.time() < deadline:
            # A posted video link on the current (success) screen is the strongest signal.
            try:
                link = page.locator(_VIDEO_HREF)
                if link.count() and link.first.is_visible():
                    href = link.first.get_attribute("href")
                    if href:
                        return self._normalize(href)
            except Exception:  # noqa: BLE001
                pass
            # A redirect to the content manager, or an explicit success toast, also means "done".
            if "/content" in page.url or self._has_text(page, _POSTED_TEXT):
                break
            time.sleep(2.0)
        # Post is finished now -- safe to consult the content list for the newest video's link.
        try:
            if "/content" not in page.url:
                page.goto(CONTENT_URL)
            link = page.locator(_VIDEO_HREF)
            link.first.wait_for(timeout=self.nav_timeout_ms)
            href = link.first.get_attribute("href")
            if href:
                return self._normalize(href)
        except Exception:  # noqa: BLE001
            pass
        return None, None

    @staticmethod
    def _has_text(page, pattern: re.Pattern) -> bool:
        try:
            return page.get_by_text(pattern).count() > 0
        except Exception:  # noqa: BLE001
            return False

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
        # Auto-dismiss any dialog (notably the "are you sure you want to leave?" beforeunload that the
        # premature navigation used to trigger mid content-check) so it can never block automation.
        page.on("dialog", lambda d: d.accept())
        page.goto(self.upload_url)
        self._ensure_logged_in(page)

        # 1) Hand the file to the hidden input and wait for TikTok to finish processing it.
        page.locator(_FILE_INPUT).first.set_input_files(str(video_path))
        page.locator(_REPLACE_BUTTON).first.wait_for(timeout=self.nav_timeout_ms)  # "Replace" => done
        self._pause(1.0, 2.0)
        self._dump(page, "01_processed")

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
        self._dump(page, "02_prepost")

        # 6) Post, clear any confirm/content-check modal, then WAIT for the check to finish before we
        # touch anything (the old code navigated away mid-check and aborted the post).
        post = page.locator(_POST_BUTTON).first
        post.wait_for(timeout=self.nav_timeout_ms)
        post.click()
        self._dump(page, "03_postclicked")
        self._confirm_post(page)
        print("[brainrotbot]   posted; waiting for TikTok's content check to finish ...")
        url, tiktok_id = self._await_completion(page)
        self._dump(page, "04_complete")
        return {
            "url": url,
            "tiktok_id": tiktok_id,
            "posted_at": time.time(),
            "public": self.privacy == "public",
            "captions_on": captions_on,
            "cover_set": cover_set,
        }
