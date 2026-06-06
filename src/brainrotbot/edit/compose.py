"""Step 4: assemble the upload-ready video from the narration + background clip.

Per story the earlier steps produce a silent 9:16 background clip already trimmed to the
narration length (Step 3) and a narration WAV (Step 2). This step muxes the narration onto the
background and appends a standard "follow our page" outro shot, writing data/final/<post_id>.mp4
(the file the upload step consumes).

ffmpeg does the work via the same bundled-static binary / subprocess plumbing as Step 3 -- we
reuse `_ffmpeg`, `_run` and `probe_duration` from `..video.background` rather than reimplementing
them (never call a bare `ffmpeg`: conda-forge's Windows build crashes). The outro is an optional
static asset (a 9:16 clip or image dropped in resources/); if it's unset or missing the final
video is just the narrated background -- the step never hard-fails on a missing asset.

Deferred to later commits: a soft background-music bed (README Step 5) and "Follow for part N"
splitting of long videos. Length is narration + outro (the README's >=1 min is only *preferred*).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..video.background import _ffmpeg, _run, probe_duration

# Image extensions get looped to a fixed duration (`-loop 1 -t`); anything else is a video.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# ffmpeg's `-i` banner lists each stream, e.g. "Stream #0:1: Audio: aac ...".
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*: Audio:")


def _has_audio(path: Path) -> bool:
    """True if the media file has an audio stream, read from ffmpeg's `-i` banner.

    Same banner-parse trick as `probe_duration` (no separate ffprobe). Used so the outro
    segment always feeds concat a valid audio input -- the outro's own audio when present,
    else synthesized silence.
    """
    _, stderr = _run([_ffmpeg(), "-hide_banner", "-i", str(path)], check=False)
    return bool(_AUDIO_STREAM_RE.search(stderr.decode("utf-8", "replace")))


class VideoEditor:
    """Combines a background clip + narration (+ optional outro) into the final video.

    Holds render config; geometry mirrors the Step 3 background frame so the outro matches.
    """

    def __init__(
        self,
        *,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
        crf: int = 23,
        preset: str = "veryfast",
        outro_file: str = "",
        outro_duration_sec: float = 4.0,
        audio_sample_rate: int = 44100,
    ):
        self.width, self.height, self.fps = width, height, fps
        self.crf, self.preset = crf, preset
        # Resolve the outro once: keep it only if it actually exists, so a stale/blank config
        # path silently degrades to the mux-only path instead of failing every story.
        outro = Path(outro_file) if outro_file else None
        self.outro = outro if (outro and outro.is_file()) else None
        self.outro_duration_sec = outro_duration_sec
        self.audio_sample_rate = audio_sample_rate

    def compose(self, background_path: Path, audio_path: Path, out_path: Path) -> dict:
        """Mux `audio_path` onto `background_path` (+ outro) -> `out_path`. Returns ledger meta."""
        ffmpeg = _ffmpeg()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.outro is None:
            self._mux_only(ffmpeg, background_path, audio_path, out_path)
        else:
            self._mux_with_outro(ffmpeg, background_path, audio_path, out_path)

        return {
            "path": str(out_path),
            "has_outro": self.outro is not None,
            "outro_file": str(self.outro) if self.outro else None,
            "duration_sec": round(probe_duration(out_path), 2),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def _mux_only(self, ffmpeg: str, bg: Path, audio: Path, out: Path) -> None:
        """No outro: copy the already-encoded background and add the narration as AAC.

        The narration is resampled to stereo `audio_sample_rate` (Kokoro's WAV is mono/24 kHz,
        which many players -- and TikTok -- render as *silent*), not stream-copied. -shortest
        guards against tiny length drift (the clip is already cut to the narration); +faststart
        moves the moov atom up front for clean web/preview playback.
        """
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(bg), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(self.audio_sample_rate), "-ac", "2",
            "-movflags", "+faststart", "-shortest", str(out),
        ])

    def _mux_with_outro(self, ffmpeg: str, bg: Path, audio: Path, out: Path) -> None:
        """Concat [background+narration] then [outro] in a single re-encoding filter pass.

        Both concat segments must share geometry/SAR/fps/pixfmt and audio rate/layout, so each
        is normalized first. An image outro is looped to outro_duration_sec; a video outro keeps
        its own length. The outro's audio is used if present, else anullsrc silence.
        """
        outro = self.outro
        outro_is_image = outro.suffix.lower() in _IMAGE_EXTS
        outro_in = (
            ["-loop", "1", "-t", f"{self.outro_duration_sec:.3f}", "-i", str(outro)]
            if outro_is_image else ["-i", str(outro)]
        )
        # An image has no audio stream; for a video, probe for one.
        outro_has_audio = (not outro_is_image) and _has_audio(outro)
        # Length of the outro segment, used to bound the silent track below so concat doesn't
        # see an infinite audio stream (image -> configured duration; video -> its real length).
        outro_dur = self.outro_duration_sec if outro_is_image else probe_duration(outro)

        w, h, fps, sr = self.width, self.height, self.fps, self.audio_sample_rate
        # Normalize main video (already correct size, but enforce sar/fps/format) and outro
        # video (fit-and-pad to the exact frame). aresample/aformat align the audio for concat.
        parts = [
            f"[0:v]setsar=1,fps={fps},format=yuv420p[v0]",
            f"[1:a]aresample={sr},aformat=channel_layouts=stereo[a0]",
            f"[2:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v1]",
        ]
        if outro_has_audio:
            parts.append(f"[2:a]aresample={sr},aformat=channel_layouts=stereo[a1]")
        else:
            parts.append(f"anullsrc=r={sr}:cl=stereo,atrim=duration={outro_dur:.3f}[a1]")
        parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
        filtergraph = ";".join(parts)

        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(bg), "-i", str(audio), *outro_in,
            "-filter_complex", filtergraph, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out),
        ])
