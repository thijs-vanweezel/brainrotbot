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

Step 5 adds an optional soft background-music bed (NCS instrumental, sourced upstream by
`music/ncs.py`): if `music_path` is supplied to compose(), both internal paths weave it in via
a shared filter chain (sidechain-duck the bed under the speech, then amix). The music input
gets `-stream_loop -1` so it tiles to any length; `duration=first` on the amix discards the
tail. When music is absent, the original code paths run unchanged -- including the mux-only
`-c:v copy` fast path (audio still goes through AAC). The "Follow for part N" split of long
videos is still deferred; length is narration + outro (the README's >=1 min is only *preferred*).
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
        music_volume_db: float = -15.0,
        music_duck: bool = True,
    ):
        self.width, self.height, self.fps = width, height, fps
        self.crf, self.preset = crf, preset
        # Resolve the outro once: keep it only if it actually exists, so a stale/blank config
        # path silently degrades to the mux-only path instead of failing every story.
        outro = Path(outro_file) if outro_file else None
        self.outro = outro if (outro and outro.is_file()) else None
        self.outro_duration_sec = outro_duration_sec
        self.audio_sample_rate = audio_sample_rate
        # Step 5 music defaults: bed sits at music_volume_db relative to the narration; if
        # music_duck is on, sidechaincompress further dips it whenever speech is present.
        self.music_volume_db = music_volume_db
        self.music_duck = music_duck

    def compose(self, background_path: Path, audio_path: Path, out_path: Path,
                music_path: Path | None = None) -> dict:
        """Mux `audio_path` onto `background_path` (+ outro, + music) -> `out_path`.

        `music_path` is optional: when supplied it weaves a soft (-`music_volume_db` dB)
        instrumental bed through both internal paths; when None the original Step 4 behaviour
        is preserved (mux-only keeps the `-c:v copy` fast path). Returns ledger meta.
        """
        ffmpeg = _ffmpeg()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.outro is None:
            self._mux_only(ffmpeg, background_path, audio_path, out_path, music_path)
        else:
            self._mux_with_outro(ffmpeg, background_path, audio_path, out_path, music_path)

        return {
            "path": str(out_path),
            "has_outro": self.outro is not None,
            "has_music": music_path is not None,
            "outro_file": str(self.outro) if self.outro else None,
            "duration_sec": round(probe_duration(out_path), 2),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def _build_music_branch(self, speech_label: str, music_idx: int) -> tuple[list[str], str]:
        """Filter snippets that mix `[music_idx:a]` softly under `[speech_label]`. Returns
        the new filter parts and the output label.

        The speech stream is asplit so its tail can drive the sidechain compressor (when
        ducking is on) without being consumed -- amix needs to receive the original speech.
        `duration=first` makes the mixed audio exactly as long as the speech; the looped
        music tail is discarded.
        """
        sr = self.audio_sample_rate
        vol = self.music_volume_db
        parts: list[str] = []
        if self.music_duck:
            parts.append(f"{speech_label}asplit=2[a_mix][a_key]")
            parts.append(
                f"[{music_idx}:a]volume={vol}dB,aresample={sr},"
                f"aformat=channel_layouts=stereo[bed_raw]"
            )
            # sidechaincompress: 1st input is the audio to compress (the bed), 2nd is the
            # sidechain key (the speech). Threshold/ratio/attack/release tuned for narration:
            # a low threshold so the bed dips on every word, fast attack, slow release.
            parts.append(
                "[bed_raw][a_key]sidechaincompress="
                "threshold=0.05:ratio=8:attack=5:release=400[bed]"
            )
            parts.append("[a_mix][bed]amix=inputs=2:duration=first:dropout_transition=0[a]")
        else:
            parts.append(
                f"[{music_idx}:a]volume={vol}dB,aresample={sr},"
                f"aformat=channel_layouts=stereo[bed]"
            )
            parts.append(f"{speech_label}[bed]amix=inputs=2:duration=first:dropout_transition=0[a]")
        return parts, "[a]"

    def _mux_only(self, ffmpeg: str, bg: Path, audio: Path, out: Path,
                  music: Path | None) -> None:
        """No outro: copy the already-encoded background and add the narration as AAC.

        The narration is resampled to stereo `audio_sample_rate` (Kokoro's WAV is mono/24 kHz,
        which many players -- and TikTok -- render as *silent*), not stream-copied. -shortest
        guards against tiny length drift (the clip is already cut to the narration); +faststart
        moves the moov atom up front for clean web/preview playback. With music present we
        switch to a `filter_complex` amix branch -- video still stream-copies.
        """
        if music is None:
            _run([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(bg), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(self.audio_sample_rate), "-ac", "2",
                "-movflags", "+faststart", "-shortest", str(out),
            ])
            return
        # Music path: stream-loop the bed so it tiles, then mix it under the narration. The
        # narration is normalized first (Kokoro mono/24 kHz -> stereo/44.1 kHz) before feeding
        # the music branch, so sidechaincompress sees consistent layouts.
        sr = self.audio_sample_rate
        parts = [f"[1:a]aresample={sr},aformat=channel_layouts=stereo[voice]"]
        music_parts, audio_out = self._build_music_branch("[voice]", music_idx=2)
        parts.extend(music_parts)
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(bg), "-i", str(audio),
            "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", ";".join(parts),
            "-map", "0:v:0", "-map", audio_out,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", str(out),
        ])

    def _mux_with_outro(self, ffmpeg: str, bg: Path, audio: Path, out: Path,
                        music: Path | None) -> None:
        """Concat [background+narration] then [outro] in a single re-encoding filter pass.

        Both concat segments must share geometry/SAR/fps/pixfmt and audio rate/layout, so each
        is normalized first. An image outro is looped to outro_duration_sec; a video outro keeps
        its own length. The outro's audio is used if present, else anullsrc silence. With music
        present the bed spans the *entire* timeline (narration + outro) and is mixed over the
        concat output -- one re-encode does it all.
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
        # Without music the concat output is the final audio. With music we rename the concat
        # output and feed it through the shared duck+amix branch; the music input is appended
        # after the outro, so its index is 3.
        audio_label = "[a]"
        extra_inputs: list[str] = []
        if music is None:
            parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
        else:
            parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a_speech]")
            music_parts, audio_label = self._build_music_branch("[a_speech]", music_idx=3)
            parts.extend(music_parts)
            extra_inputs = ["-stream_loop", "-1", "-i", str(music)]
        filtergraph = ";".join(parts)

        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(bg), "-i", str(audio), *outro_in, *extra_inputs,
            "-filter_complex", filtergraph, "-map", "[v]", "-map", audio_label,
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out),
        ])
