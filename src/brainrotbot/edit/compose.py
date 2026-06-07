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

import random
import re
from pathlib import Path

from ..video.background import _ffmpeg, _run, probe_duration

# Image extensions get looped to a fixed duration (`-loop 1 -t`); anything else is a video.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# ffmpeg's `-i` banner lists each stream, e.g. "Stream #0:1: Audio: aac ...".
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*: Audio:")


def _escape_subtitles_path(path: Path) -> str:
    """Escape a path for ffmpeg's `subtitles=` filter (Windows-safe).

    The filtergraph parser treats `:` as an option separator and `\\` as an escape, so on Windows a
    bare `C:\\Users\\...` breaks. Convert backslashes to forward slashes and escape the drive colon;
    the caller wraps the result in single quotes (`subtitles=filename='...'`).
    """
    return str(path).replace("\\", "/").replace(":", "\\:")


def _has_audio(path: Path) -> bool:
    """True if the media file has an audio stream, read from ffmpeg's `-i` banner.

    Same banner-parse trick as `probe_duration` (no separate ffprobe). Used so the outro
    segment always feeds concat a valid audio input -- the outro's own audio when present,
    else synthesized silence.
    """
    _, stderr = _run([_ffmpeg(), "-hide_banner", "-i", str(path)], check=False)
    return bool(_AUDIO_STREAM_RE.search(stderr.decode("utf-8", "replace")))


def pick_music_start(music_dur: float, target_dur: float, intro_skip_sec: float = 5.0,
                     rng: random.Random | None = None) -> float:
    """Random start offset into the music track for the soft bed.

    Mirrors `video/background.py`'s window picker: if the track is at least as long as the
    needed `target_dur` (narration + outro), pick uniformly in [intro_skip, music_dur-target];
    the `intro_skip_sec` floor skips NCS's frequent quiet intro/buildup. If the track is too
    short, start at 0 and rely on -stream_loop to fill -- the loop seam will be audible but
    that's the only honest option. `rng` is injectable for deterministic tests.
    """
    if music_dur <= target_dur + 0.5:
        return 0.0
    hi = music_dur - target_dur
    lo = min(intro_skip_sec, hi)
    return (rng or random).uniform(lo, hi)


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
        music_intro_skip_sec: float = 5.0,
        voice_volume_db: float = 0.0,
        output_volume_db: float = 0.0,
        subtitle_fonts_dir: str = "",
    ):
        self.width, self.height, self.fps = width, height, fps
        self.crf, self.preset = crf, preset
        # Directory libass loads caption fonts from (Step 4.5). The .ass references the font by
        # family name (e.g. "Anton"); fontsdir points libass at resources/ so it finds the bundled
        # Anton-Regular.ttf without relying on a system-wide install. "" -> rely on system fonts.
        self.subtitle_fonts_dir = subtitle_fonts_dir
        # Resolve the outro once: keep it only if it actually exists, so a stale/blank config
        # path silently degrades to the mux-only path instead of failing every story.
        outro = Path(outro_file) if outro_file else None
        self.outro = outro if (outro and outro.is_file()) else None
        self.outro_duration_sec = outro_duration_sec
        self.audio_sample_rate = audio_sample_rate
        # Step 5 music defaults: bed sits at music_volume_db relative to the narration; if
        # music_duck is on, sidechaincompress further dips it whenever speech is present.
        # music_intro_skip_sec is the floor on the random start offset into the track --
        # mirrors video/[video].intro_skip_sec, since NCS tracks usually have a quiet intro.
        self.music_volume_db = music_volume_db
        self.music_duck = music_duck
        self.music_intro_skip_sec = music_intro_skip_sec
        # voice_volume_db is an absolute gain on the narration alone (applied wherever the voice is
        # normalized, in every path) -- the simplest way to make speech louder, and it also offsets
        # amix's 1/n attenuation when a music bed is present. output_volume_db is an overall gain on
        # the *finished* mix (voice + music + outro); it's followed by a brick-wall limiter so a
        # boost raises loudness without clipping. Both default to 0 (no change).
        self.voice_volume_db = voice_volume_db
        self.output_volume_db = output_volume_db

    def _voice_gain(self) -> str:
        """Chainable filter suffix (leading comma) that boosts the narration, or "" if no boost."""
        return f",volume={self.voice_volume_db}dB" if self.voice_volume_db else ""

    def _output_gain(self, parts: list[str], label: str) -> str:
        """Append a final overall-gain + peak-limiter stage to `parts`; return the label to map.

        The limiter (brick-wall at ~0.95 full-scale) keeps a positive overall boost from clipping.
        No-op when output_volume_db is 0 -- the incoming label is returned unchanged.
        """
        if not self.output_volume_db:
            return label
        parts.append(f"{label}volume={self.output_volume_db}dB,alimiter=limit=0.95[aout]")
        return "[aout]"

    def _subtitles_filter(self, subtitle_path: Path) -> str:
        """The `subtitles=` filter snippet that burns `subtitle_path`, with fontsdir if configured."""
        flt = f"subtitles=filename='{_escape_subtitles_path(subtitle_path)}'"
        if self.subtitle_fonts_dir:
            flt += f":fontsdir='{_escape_subtitles_path(Path(self.subtitle_fonts_dir))}'"
        return flt

    def compose(self, background_path: Path, audio_path: Path, out_path: Path,
                music_path: Path | None = None, subtitle_path: Path | None = None) -> dict:
        """Mux `audio_path` onto `background_path` (+ outro, + music, + burned-in captions) -> `out_path`.

        `music_path` and `subtitle_path` are both optional. Music weaves a soft (-`music_volume_db`
        dB) instrumental bed through both internal paths. Captions (Step 4.5) are burned over the
        narration/background segment only -- never the outro. With no music and no subtitles the
        mux-only path keeps its `-c:v copy` fast path; burning captions there forces a video
        re-encode (libass can't draw onto a stream-copied frame). Returns ledger meta.
        """
        ffmpeg = _ffmpeg()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Random window into the music track so each video uses a different slice. Mirrors
        # the video step's random offset. Probe both the music and the *target* (background +
        # any outro) so we can keep the window from running off the end of the track.
        music_start = 0.0
        if music_path is not None:
            music_dur = probe_duration(music_path)
            outro_dur = self._outro_duration() if self.outro is not None else 0.0
            target = probe_duration(background_path) + outro_dur
            music_start = pick_music_start(music_dur, target, self.music_intro_skip_sec)

        if self.outro is None:
            self._mux_only(ffmpeg, background_path, audio_path, out_path, music_path, music_start,
                           subtitle_path)
        else:
            self._mux_with_outro(ffmpeg, background_path, audio_path, out_path, music_path,
                                 music_start, subtitle_path)

        return {
            "path": str(out_path),
            "has_outro": self.outro is not None,
            "has_music": music_path is not None,
            "has_subtitles": subtitle_path is not None,
            "music_start_sec": round(music_start, 2) if music_path is not None else None,
            "outro_file": str(self.outro) if self.outro else None,
            "duration_sec": round(probe_duration(out_path), 2),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def _outro_duration(self) -> float:
        """Length of the outro segment in the final video (image -> configured; video -> real).
        Hoisted out of `_mux_with_outro` so `compose()` can reuse it for the music window."""
        if self.outro is None:
            return 0.0
        if self.outro.suffix.lower() in _IMAGE_EXTS:
            return self.outro_duration_sec
        return probe_duration(self.outro)

    def _build_music_branch(self, speech_label: str, music_idx: int) -> tuple[list[str], str]:
        """Filter snippets that mix `[music_idx:a]` softly under `[speech_label]`. Returns
        the new filter parts and the output label.

        The speech stream is asplit so its tail can drive the sidechain compressor (when
        ducking is on) without being consumed -- amix needs to receive the original speech.
        `duration=first` makes the mixed audio exactly as long as the speech; the looped
        music tail is discarded. `normalize=0` is important: amix's default 1/n normalisation
        would halve (-6 dB) the narration just because a bed is present -- disabling it keeps the
        voice at full level (the bed is already soft via music_volume_db, and the output limiter
        guards peaks), so the voice is as loud with music as without.
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
            parts.append("[a_mix][bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
        else:
            parts.append(
                f"[{music_idx}:a]volume={vol}dB,aresample={sr},"
                f"aformat=channel_layouts=stereo[bed]"
            )
            parts.append(f"{speech_label}[bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
        return parts, "[a]"

    def _mux_only(self, ffmpeg: str, bg: Path, audio: Path, out: Path,
                  music: Path | None, music_start: float = 0.0,
                  subtitle_path: Path | None = None) -> None:
        """No outro: add the narration (+ optional music bed, + optional burned-in captions).

        The narration is resampled to stereo `audio_sample_rate` (Kokoro's WAV is mono/24 kHz,
        which many players -- and TikTok -- render as *silent*), not stream-copied. -shortest
        guards against tiny length drift (the clip is already cut to the narration); +faststart
        moves the moov atom up front for clean web/preview playback.

        Four cases: plain mux keeps the fast `-c:v copy` path; music alone adds an audio-only
        `filter_complex` (video still copies); captions force a libx264 re-encode (libass can't
        draw onto a copied stream) with a `[0:v]subtitles=...[v]` branch; music + captions share
        one `filter_complex` carrying both the video and audio branches.
        """
        sr = self.audio_sample_rate
        sub = subtitle_path is not None

        # --- fast path: no music, no captions -> stream-copy the video ---
        if music is None and not sub:
            # The only audio is the narration, so voice and overall gains both act on it; apply
            # them via -filter:a (video still stream-copies). With no gains there's no audio filter
            # at all -- the original fast path. The limiter only follows a positive overall boost.
            af = []
            if self.voice_volume_db:
                af.append(f"volume={self.voice_volume_db}dB")
            if self.output_volume_db:
                af.append(f"volume={self.output_volume_db}dB")
                af.append("alimiter=limit=0.95")
            af_opt = ["-filter:a", ",".join(af)] if af else []
            _run([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(bg), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *af_opt,
                "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
                "-movflags", "+faststart", "-shortest", str(out),
            ])
            return

        # Build the (optional) video subtitle branch and the audio branch into one filtergraph.
        parts: list[str] = []
        video_map = "0:v:0"
        if sub:
            parts.append(f"[0:v]{self._subtitles_filter(subtitle_path)},"
                         f"fps={self.fps},format=yuv420p[v]")
            video_map = "[v]"

        inputs = ["-i", str(bg), "-i", str(audio)]
        audio_codec = ["-c:a", "aac", "-b:a", "192k"]
        if music is None:
            # Captions only: normalize the narration (+ voice gain) through the graph, no bed.
            parts.append(f"[1:a]aresample={sr},aformat=channel_layouts=stereo{self._voice_gain()}[voice]")
            audio_map = "[voice]"
        else:
            # Music (+ maybe captions): normalize+boost narration then duck+amix the looped bed.
            parts.append(f"[1:a]aresample={sr},aformat=channel_layouts=stereo{self._voice_gain()}[voice]")
            music_parts, audio_map = self._build_music_branch("[voice]", music_idx=2)
            parts.extend(music_parts)
            # -ss before -i is the input seek (fast); each loop iteration restarts at this offset.
            inputs += ["-ss", f"{music_start:.3f}", "-stream_loop", "-1", "-i", str(music)]
        # Overall boost (+ limiter) on the finished mix, common to both branches.
        audio_map = self._output_gain(parts, audio_map)

        video_codec = (["-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
                        "-pix_fmt", "yuv420p"] if sub else ["-c:v", "copy"])
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex", ";".join(parts),
            "-map", video_map, "-map", audio_map, *video_codec, *audio_codec,
            "-movflags", "+faststart", "-shortest", str(out),
        ])

    def _mux_with_outro(self, ffmpeg: str, bg: Path, audio: Path, out: Path,
                        music: Path | None, music_start: float = 0.0,
                        subtitle_path: Path | None = None) -> None:
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
        # Burn captions onto the narration/background segment ([0:v]) only -- the outro ([2:v]) is
        # deliberately left clean. Prepended before the normalize filters so libass draws first.
        sub_prefix = f"{self._subtitles_filter(subtitle_path)}," if subtitle_path is not None else ""
        # Normalize main video (already correct size, but enforce sar/fps/format) and outro
        # video (fit-and-pad to the exact frame). aresample/aformat align the audio for concat.
        parts = [
            f"[0:v]{sub_prefix}setsar=1,fps={fps},format=yuv420p[v0]",
            f"[1:a]aresample={sr},aformat=channel_layouts=stereo{self._voice_gain()}[a0]",
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
            extra_inputs = ["-ss", f"{music_start:.3f}", "-stream_loop", "-1", "-i", str(music)]
        # Overall boost (+ limiter) on the finished voice+outro(+music) timeline.
        audio_label = self._output_gain(parts, audio_label)
        filtergraph = ";".join(parts)

        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(bg), "-i", str(audio), *outro_in, *extra_inputs,
            "-filter_complex", filtergraph, "-map", "[v]", "-map", audio_label,
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out),
        ])
