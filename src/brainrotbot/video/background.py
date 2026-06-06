"""Step 3: build a vertical background clip for a story, sized to its narration.

Pipeline per story:
  1. Pick a source from the curated `[video].sources` pool (round-robin -> A/B testing),
     download it once with yt-dlp and cache it on disk (data/video_cache/), reused across runs.
  2. Find the most *active* window of the source (length = the narration's duration) by
     scoring motion across the source's keyframes (decoded tiny + grayscale) in numpy.
  3. Trim to that window and center-crop the 16:9 gameplay to a 9:16 TikTok frame with ffmpeg.
Output: data/video/<post_id>.mp4 (silent; the narration is muxed on in the Step 4 edit).

ffmpeg (bundled by the `imageio-ffmpeg` wheel -- a static build, so no system install / no
Windows DLL hell) and yt-dlp do the heavy lifting via subprocess; numpy is the only other heavy
dep, imported lazily, so the package stays importable for the test suite without the `[video]`
extra. Duration is read from `ffmpeg -i` (no separate ffprobe needed). The pure selection math
(`pick_source`, `best_window_start`) lives here too and is unit-tested without any of those tools.
"""

from __future__ import annotations

import hashlib
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


def best_window_start(motion: list[float], sample_fps: float, window_sec: float) -> tuple[float, float]:
    """Pick the highest-motion contiguous window from a per-frame motion series.

    `motion[i]` is the motion energy between sampled frames i and i+1 (frames sampled at
    `sample_fps`). Returns (start_sec, score) where the window of length `window_sec` starting
    at start_sec has the greatest summed motion. A clip shorter than the window -> (0.0, sum).
    Pure (no IO) so it is unit-testable; `score` is logged for the analytics ledger.
    """
    win = max(1, round(window_sec * sample_fps))
    # prefix[k] = sum(motion[:k]); window sum = prefix[i+win]-prefix[i].
    prefix = [0.0]
    for m in motion:
        prefix.append(prefix[-1] + m)
    if win >= len(motion):
        return 0.0, prefix[-1]
    best_i, best_sum = 0, -1.0
    for i in range(len(motion) - win + 1):
        s = prefix[i + win] - prefix[i]
        if s > best_sum:
            best_sum, best_i = s, i
    return best_i / sample_fps, best_sum


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


def ensure_cached(url: str, cache_dir: Path, *, max_height: int) -> Path:
    """Download `url` to the cache once (keyed by URL hash) and return the local file.

    Already-cached sources are reused, so the network/disk cost is paid a single time even
    though many stories share the pool. Video-only is fetched (no audio) -- the background is silent.
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
    _run([
        ytdlp, "--no-playlist", "--no-progress", "-f", fmt,
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
    """Turns a curated source pool into per-story vertical clips. Holds render/sampling config;
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
        sample_width: int = 64,
        sample_height: int = 36,
    ):
        self.cache_dir = Path(cache_dir)
        self.width, self.height, self.fps = width, height, fps
        self.crf, self.preset, self.max_source_height = crf, preset, max_source_height
        self.sample_width, self.sample_height = sample_width, sample_height

    def _sample_motion(self, src: Path, src_dur: float) -> tuple[list[float], float]:
        """Score motion across `src` from its keyframes; return (motion, effective_fps).

        `-skip_frame nokey` makes the decoder skip P/B frames, so even a multi-GB source is
        sampled in seconds (decoding every frame would be minutes). Keyframes are ~evenly spaced,
        so we treat them as a uniform series at effective_fps = keyframes / duration. Frames are
        downscaled hard (e.g. 64x36) so the whole series fits in memory; the per-frame mean
        absolute difference is a cheap, robust proxy for on-screen action.
        """
        import numpy as np  # heavy + optional; imported on first use (mirrors the TTS module)

        w, h = self.sample_width, self.sample_height
        raw, _ = _run([
            _ffmpeg(), "-hide_banner", "-loglevel", "error",
            "-skip_frame", "nokey", "-i", str(src),
            "-an", "-vf", f"scale={w}:{h},format=gray", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ], capture=True)
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)
        if len(frames) < 2 or src_dur <= 0:
            return [], 0.0
        motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)).tolist()
        return motion, len(frames) / src_dur

    def make(self, source_url: str, duration_sec: float, out_path: Path) -> dict:
        """Produce a `duration_sec`-long 9:16 clip from `source_url` at `out_path`.

        Returns metadata for the ledger (source, chosen window, motion score, geometry).
        """
        ffmpeg = _ffmpeg()
        src = ensure_cached(source_url, self.cache_dir, max_height=self.max_source_height)
        src_dur = probe_duration(src)

        # Source too short for the narration -> start at 0 and loop it to fill; else pick the
        # most active window by motion. (+0.5s slack avoids a near-equal-length corner case.)
        if src_dur <= duration_sec + 0.5:
            start, score, looped = 0.0, None, True
        else:
            motion, eff_fps = self._sample_motion(src, src_dur)
            start, score = best_window_start(motion, eff_fps, duration_sec) if motion else (0.0, None)
            looped = False

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
            "motion_score": round(score, 4) if score is not None else None,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
