"""Step 3: build a vertical background clip for a story, sized to its narration.

Pipeline per story:
  1. Pick a source from the curated `[video].sources` pool (round-robin -> A/B testing),
     download it once with yt-dlp and cache it on disk (data/video_cache/), reused across runs.
  2. Trim a window the length of the narration from a random offset in the source (for variety),
     and center-crop the 16:9 gameplay to a 9:16 TikTok frame with ffmpeg.
Output: data/video/<post_id>.mp4 (silent; the narration is muxed on in the Step 4 edit).

ffmpeg (bundled by the `imageio-ffmpeg` wheel -- a static build, so no system install / no
Windows DLL hell) and yt-dlp do the heavy lifting via subprocess. Duration is read from
`ffmpeg -i` (no separate ffprobe needed). The pure source-rotation helper (`pick_source`) lives
here too and is unit-tested without any of those tools.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import subprocess
from pathlib import Path

# ffmpeg's stderr banner line, e.g. "  Duration: 00:10:23.45, start: ...".
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


def pick_source(sources: list[str], index: int) -> str:
    """Round-robin a story (by its position in the run) over the source-video pool.

    Mirrors tts.pick_voice: deterministic so rotation is reproducible/testable, and the
    chosen source is logged per story to feed the Step 6 A/B analytics (which background works).
    """
    if not sources:
        raise ValueError("source pool is empty")
    return sources[index % len(sources)]


def _ffmpeg() -> str:
    """Path to a working ffmpeg: the static one bundled by imageio-ffmpeg, else PATH.

    Bundled-first is deliberate -- conda-forge's Windows ffmpeg build is prone to DLL-entrypoint
    crashes, so we don't trust a PATH ffmpeg unless imageio-ffmpeg is unavailable.
    """
    try:
        import imageio_ffmpeg  # optional [video] dep; imported lazily

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("ffmpeg not found -- install it (pip install imageio-ffmpeg).")
        return found


def _require(binary: str) -> str:
    """Resolve an external tool on PATH or raise an actionable error."""
    found = shutil.which(binary)
    if not found:
        raise RuntimeError(f"'{binary}' not found on PATH -- install it (pip install {binary}).")
    return found


def _run(cmd: list[str], *, capture: bool = False, check: bool = True) -> tuple[bytes, bytes]:
    """Run a subprocess; return (stdout, stderr). Raise on non-zero unless check=False."""
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-800:]
        raise RuntimeError(f"command failed ({Path(cmd[0]).name}, code {proc.returncode}): {err}")
    return proc.stdout or b"", proc.stderr or b""


def ensure_cached(
    url: str, cache_dir: Path, *,
    max_height: int, cookies_from_browser: str = "", cookies_file: str = "",
) -> Path:
    """Download `url` to the cache once (keyed by URL hash) and return the local file.

    Already-cached sources are reused, so the network/disk cost is paid a single time even
    though many stories share the pool. Video-only is fetched (no audio) -- the background is silent.
    To clear YouTube's anti-bot gate, pass either `cookies_file` (a Netscape cookies.txt, most
    reliable on Windows) or `cookies_from_browser` (e.g. "edge"/"chrome"; needs the browser closed,
    as a running browser locks its cookie DB). `cookies_file` wins if both are set.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    hit = [p for p in cache_dir.glob(f"{key}.*") if p.suffix != ".part"]
    if hit:
        return hit[0]
    ytdlp = _require("yt-dlp")
    # Prefer a single H.264 video-only stream: universally decodable and needs no ffmpeg merge
    # (we drop audio at render anyway). Fall back to any video-only, then a progressive stream.
    fmt = (
        f"bestvideo[height<={max_height}][vcodec^=avc1]/bestvideo[height<={max_height}]"
        f"/best[height<={max_height}]/best"
    )
    # YouTube increasingly gates anonymous downloads behind a "confirm you're not a bot" check;
    # logged-in cookies clear it. A cookies.txt file is preferred (robust on Windows); else a
    # browser. Off unless one is configured.
    if cookies_file:
        cookie_args = ["--cookies", cookies_file]
    elif cookies_from_browser:
        cookie_args = ["--cookies-from-browser", cookies_from_browser]
    else:
        cookie_args = []
    # YouTube's "n-signature" throttling challenge now needs yt-dlp's external JS solver scripts
    # (EJS), fetched + cached from GitHub on first use; without them only image formats are offered.
    _run([
        ytdlp, "--no-playlist", "--no-progress", "-f", fmt, *cookie_args,
        "--remote-components", "ejs:github",
        "--ffmpeg-location", _ffmpeg(), "-o", str(cache_dir / f"{key}.%(ext)s"), url,
    ])
    hit = [p for p in cache_dir.glob(f"{key}.*") if p.suffix != ".part"]
    if not hit:
        raise RuntimeError(f"yt-dlp produced no file for {url}")
    return hit[0]


def probe_duration(path: Path) -> float:
    """Return a media file's duration in seconds, parsed from ffmpeg's banner.

    `ffmpeg -i <file>` (no output) exits non-zero by design but prints "Duration: HH:MM:SS.ss"
    to stderr, so we read it there -- avoiding a separate ffprobe binary.
    """
    _, stderr = _run([_ffmpeg(), "-hide_banner", "-i", str(path)], check=False)
    m = _DURATION_RE.search(stderr.decode("utf-8", "replace"))
    if not m:
        raise RuntimeError(f"could not read duration from {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


class BackgroundVideoMaker:
    """Turns a curated source pool into per-story vertical clips. Holds render config;
    no in-memory caching is needed since downloaded sources are cached on disk across runs."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
        crf: int = 23,
        preset: str = "veryfast",
        max_source_height: int = 1080,
        intro_skip_sec: float = 5.0,
        cookies_from_browser: str = "",
        cookies_file: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.width, self.height, self.fps = width, height, fps
        self.crf, self.preset, self.max_source_height = crf, preset, max_source_height
        self.intro_skip_sec = intro_skip_sec
        self.cookies_from_browser, self.cookies_file = cookies_from_browser, cookies_file

    def make(self, source_url: str, duration_sec: float, out_path: Path) -> dict:
        """Produce a `duration_sec`-long 9:16 clip from `source_url` at `out_path`.

        Returns metadata for the ledger (source, chosen window, geometry).
        """
        ffmpeg = _ffmpeg()
        src = ensure_cached(
            source_url, self.cache_dir, max_height=self.max_source_height,
            cookies_from_browser=self.cookies_from_browser, cookies_file=self.cookies_file,
        )
        src_dur = probe_duration(src)

        # Source too short for the narration -> start at 0 and loop it to fill; else trim a
        # window from a random offset (variety across videos). +0.5s slack avoids a
        # near-equal-length corner case. The window starts no earlier than intro_skip_sec to
        # avoid the source's intro/title card -- clamped down when the source is barely long enough.
        if src_dur <= duration_sec + 0.5:
            start, looped = 0.0, True
        else:
            hi = src_dur - duration_sec
            start, looped = random.uniform(min(self.intro_skip_sec, hi), hi), False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        loop_args = ["-stream_loop", "-1"] if looped else []
        # crop=min(iw, ih*9/16):ih centers a 9:16 column (guarding already-vertical sources),
        # then scale to the exact TikTok frame. -an: the narration is added in the Step 4 edit.
        vf = f"crop='min(iw,ih*{self.width}/{self.height})':ih,scale={self.width}:{self.height},fps={self.fps}"
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            *loop_args, "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration_sec:.3f}",
            "-vf", vf, "-an", "-c:v", "libx264", "-crf", str(self.crf),
            "-preset", self.preset, "-pix_fmt", "yuv420p", str(out_path),
        ])
        return {
            "path": str(out_path),
            "source_url": source_url,
            "source_id": src.stem,
            "start_sec": round(start, 2),
            "duration_sec": round(duration_sec, 2),
            "looped": looped,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
