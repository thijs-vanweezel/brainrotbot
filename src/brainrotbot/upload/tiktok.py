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

# --- Studio selectors. The data-e2e ones were pinned from a real DOM dump (data/upload_debug) and are
# stable; the rest are best-effort. NOTE: `button:has-text('Post')` is WRONG here -- it also matches the
# "Posts" sidebar nav, and clicking that navigates away (the "are you sure you want to exit?" modal). Use
# the data-e2e id. --------------------------------------------------------------------------------------
_FILE_INPUT = "input[type=file]"                       # hidden file picker on the upload page
_CAPTION_EDITOR = "div[contenteditable='true']"        # DraftJS caption box
_POST_BUTTON = "[data-e2e=post_video_button]"          # the real submit (NOT the "Posts" nav button)
_REPLACE_BUTTON = "button:has-text('Replace')"         # appears once a video finished processing
_VIDEO_HREF = "a[href*='/video/']"                     # links that carry a posted video id
_VIDEO_ID_RE = re.compile(r"/video/(\d+)")
# A real "the post finished" signal -- only trust these, not the upload page's ambient nav text.
_POSTED_TEXT = re.compile(r"your video has been (uploaded|posted)|posted to|view profile", re.I)
# In-page "Are you sure you want to exit?" modal (React, NOT a native dialog) -- its stay/leave buttons.
_EXIT_CANCEL = re.compile(r"^cancel$", re.I)
# The "Content check lite" toggle (pinned from a DOM dump): its label sits in a `headline-wrapper`, the
# switch in the sibling `headline-switch`. Scoping by the label text avoids the identical HD-uploads
# switch. It defaults ON and re-arms each upload, so we flip it off every time (see _disable_content_check).
_SHOW_MORE = re.compile(r"^show more$", re.I)  # expands the advanced settings section if collapsed
# The Content Check Lite control lives in the `.card` whose text contains "Content check lite"; its switch
# is a hidden <input role=switch>, and aria-checked=true (on a child) means the check is ON. A force-click
# on the input is the only thing that actually flips it (React ignores a programmatic .click() on the root).
_CHECK_CARD = "Content check lite"


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

    def _set_cover(self, page, cover_path: Path) -> bool:
        """Best-effort custom cover from the Step 6 PNG. Returns True on success.

        Open the cover editor and set its file `input` DIRECTLY -- we must NOT click the dialog's
        "Upload" button, which fires the native OS file picker (a real dialog Playwright can't close, so
        it stalls the run until dismissed by hand). A page-level filechooser handler (registered in
        upload()) is a backstop that fills any stray picker. Then click the dialog's "Save".
        """
        try:
            opened = False
            for label in ("Edit cover", "Select cover"):
                btn = page.get_by_text(label, exact=False)
                if btn.count() and btn.first.is_visible():
                    btn.first.click()
                    opened = True
                    break
            if not opened:
                return False
            dialog = page.get_by_role("dialog")
            dialog.first.wait_for(state="visible", timeout=10000)  # cover editor opened
            # Set the cover file straight onto the dialog's input (no "Upload" click -> no OS dialog).
            inputs = page.locator(_FILE_INPUT)
            inputs.nth(inputs.count() - 1).set_input_files(str(cover_path))
            self._pause(1.5, 2.5)  # let the preview render
            # Click Save, then VERIFY the modal actually closed. The intermittent "stuck on the cover
            # window" is a Save click that didn't register -- the modal stays open and blocks Post. Retry
            # once, then force-dismiss so the run never hangs (degrades to TikTok's auto frame).
            save = page.get_by_role("button", name=re.compile(r"^(save|confirm|done|apply)$", re.I))
            for _ in range(2):
                if save.count():
                    try:
                        save.first.click(timeout=8000)
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    dialog.first.wait_for(state="hidden", timeout=6000)  # closed/detached
                    return True
                except Exception:  # noqa: BLE001 -- still open: loop retries the Save click once
                    self._pause(0.5, 1.0)
            self._close_any_dialog(page)
            print("[brainrotbot]   (cover dialog wouldn't close -- using auto frame)")
            return False
        except Exception as exc:  # noqa: BLE001 -- fall back to TikTok's auto frame
            print(f"[brainrotbot]   (custom cover not set, using auto frame: {exc})")
            self._close_any_dialog(page)
            return False

    def _close_any_dialog(self, page) -> None:
        """Dismiss a lingering modal (e.g. the cover editor) so it can't block the Post step.

        Prefer the modal's own Cancel; fall back to Escape. Safe no-op when nothing is open.
        """
        try:
            dlg = page.get_by_role("dialog")
            if dlg.count() and dlg.first.is_visible():
                cancel = page.get_by_role("button", name=re.compile(r"^cancel$", re.I))
                if cancel.count() and cancel.first.is_visible():
                    cancel.first.click()
                else:
                    page.keyboard.press("Escape")
                self._pause()
        except Exception:  # noqa: BLE001
            pass

    def _disable_content_check(self, page) -> bool:
        """Flip 'Content check lite' OFF so the post completes immediately.

        That toggle defaults ON and re-arms every upload; left on it runs a ~10 min pre-post eligibility
        check that spawns confirmation/exit modals and stalls automation. We turn it off each time. The
        "Checks" card renders a little after the video finishes processing, so we wait for the switch
        (and expand the advanced "Show more" section if it's collapsed). Best-effort; returns True when
        the check is off (or already was).
        """
        for _ in range(3):
            try:
                card = page.locator(".card").filter(has_text=_CHECK_CARD).last
                inp = card.locator("input[role=switch]")
                inp.first.wait_for(state="attached", timeout=8000)  # input is hidden -> attached, not visible
                if card.locator("[aria-checked='true']").count() == 0:
                    return True  # already off
                inp.first.click(force=True)  # force-click the hidden input (the only thing React honours)
                self._pause()
                if card.locator("[aria-checked='true']").count() == 0:
                    print("[brainrotbot]   Content Check Lite -> OFF")
                else:
                    print("[brainrotbot]   (Content Check Lite click did not register)")
                return True
            except Exception:  # noqa: BLE001
                pass
            try:  # card may be collapsed under advanced settings -- expand and retry
                more = page.get_by_text(_SHOW_MORE)
                if more.count() and more.first.is_visible():
                    more.first.click()
            except Exception:  # noqa: BLE001
                pass
            self._pause(1.0, 2.0)
        print("[brainrotbot]   (Content Check Lite toggle not found -- the ~10 min check may still run)")
        return False

    def _dismiss_caption_popup(self, page) -> None:
        """Close the hashtag/mention suggestion dropdown after typing the caption.

        TikTok pops a suggestion overlay while typing '#fyp' etc.; left open it covers the form and the
        next actions (cover, visibility) get stuck scrolling around it. Press Escape, then click a
        neutral static label ('Details' heading) to blur the editor -- the same "click anywhere" dismissal
        a human would do.
        """
        try:
            page.keyboard.press("Escape")
            self._pause(0.3, 0.6)
            neutral = page.get_by_text("Details", exact=True)
            if neutral.count() and neutral.first.is_visible():
                neutral.first.click()
        except Exception:  # noqa: BLE001
            pass
        self._pause()

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

    def _stay(self, page) -> None:
        """If TikTok's in-page 'Are you sure you want to exit?' modal appears, click Cancel to stay.

        This modal is a React element (not a native dialog, so the page.on('dialog') handler can't catch
        it). It only shows when something tries to leave the upload page -- we now never navigate until the
        post is confirmed, but this is a defensive backstop so a stray modal can't strand the post.
        """
        try:
            btn = page.get_by_role("button", name=_EXIT_CANCEL)
            if btn.count() and btn.first.is_visible():
                btn.first.click()
        except Exception:  # noqa: BLE001
            pass

    def _await_completion(self, page) -> tuple[str | None, str | None]:
        """Block until the post truly lands; return (url, id), or (None, None) if never confirmed.

        Fix for the original bug: we must NOT navigate away or close while TikTok is still posting (that
        triggered the exit modal and aborted the post). Poll up to completion_timeout_sec for a real
        "posted" signal -- a visible /video/ link, a success toast, or a redirect to the content manager --
        and ONLY navigate to the content list once we're sure the post finished. If it never confirms we
        return (None, None) WITHOUT navigating, so the caller marks it unconfirmed and keeps the media.
        (With Content Check Lite OFF this resolves in seconds; with it ON, the check can take ~10 min.)
        """
        deadline = time.time() + self.completion_timeout_sec
        completed = False
        while time.time() < deadline:
            try:  # strongest signal: a posted /video/ link is on screen
                link = page.locator(_VIDEO_HREF)
                if link.count() and link.first.is_visible():
                    href = link.first.get_attribute("href")
                    if href:
                        return self._normalize(href)
            except Exception:  # noqa: BLE001
                pass
            if "/content" in page.url or self._has_text(page, _POSTED_TEXT):
                completed = True
                break
            self._stay(page)  # defensive: never let an exit modal abandon the post
            time.sleep(2.0)
        if not completed:
            return None, None  # don't navigate mid-upload (avoids the exit modal) -> caller: unconfirmed
        # Post landed -- now it's safe to read the newest video's link from the content manager.
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

        Raises on hard failures (no session, processing never completed) so the queue marks the video
        failed. Cosmetic steps (cover, visibility) degrade quietly. Subtitles need no action: TikTok now
        auto-captions every video (American English + Japanese), and removed the upload-time toggle -- so
        `captions_on` reflects that automatic default rather than a UI click.
        """
        if self._ctx is None:
            raise RuntimeError("TikTokUploader used outside its context manager (`with uploader:`)")
        page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        # Auto-dismiss any dialog (notably the "are you sure you want to leave?" beforeunload that the
        # premature navigation used to trigger mid content-check) so it can never block automation.
        page.on("dialog", lambda d: d.accept())
        # Backstop: if any click ever opens the native OS file picker (e.g. the cover dialog's "Upload"
        # button), fill it with the cover image instead of leaving a real dialog blocking the run.
        if cover_path and self.set_cover:
            page.on("filechooser", lambda fc: fc.set_files(str(cover_path)))
        page.goto(self.upload_url)
        self._ensure_logged_in(page)

        # 1) Hand the file to the hidden input and wait for TikTok to finish processing it.
        page.locator(_FILE_INPUT).first.set_input_files(str(video_path))
        page.locator(_REPLACE_BUTTON).first.wait_for(timeout=self.nav_timeout_ms)  # "Replace" => done
        self._pause(1.0, 2.0)
        self._dump(page, "01_processed")

        # 2) Caption = title + hashtags. Clear the placeholder text, then type, then DISMISS the hashtag
        # suggestion dropdown -- left open it overlays the form and our next clicks fight it (scroll loop).
        editor = page.locator(_CAPTION_EDITOR).first
        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Delete")
        editor.type(caption, delay=15)
        self._dismiss_caption_popup(page)

        # 3) Subtitles: nothing to toggle -- TikTok auto-captions every video. 4) cover, 5) visibility.
        captions_on = self.subtitles  # TikTok auto-captions by default; we just record the intent
        cover_set = bool(cover_path) and self.set_cover and self._set_cover(page, Path(cover_path))
        self._set_public(page)
        # Turn OFF Content Check Lite now that the "Checks" card has rendered, so the post lands
        # immediately instead of waiting on the ~10 min pre-post check.
        check_off = self._disable_content_check(page)
        self._dump(page, "02_prepost")

        # 6) Click the REAL post button (data-e2e, NOT the "Posts" nav), then wait for the post to land
        # before touching anything (the old code clicked the nav / navigated away and aborted the post).
        self._close_any_dialog(page)  # never let a leftover modal (e.g. the cover editor) block Post
        post = page.locator(_POST_BUTTON).first
        post.wait_for(timeout=self.nav_timeout_ms)
        post.click()
        self._dump(page, "03_postclicked")
        print("[brainrotbot]   posted; waiting for it to finish (Content Check Lite, if on, can take ~10 min) ...")
        url, tiktok_id = self._await_completion(page)
        self._dump(page, "04_complete")
        return {
            "url": url,
            "tiktok_id": tiktok_id,
            "posted_at": time.time(),
            "public": self.privacy == "public",
            "captions_on": captions_on,
            "cover_set": cover_set,
            "content_check_off": check_off,
        }
