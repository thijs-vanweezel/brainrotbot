"""Lay a "blur" censor SFX over banned-word intervals in the narration WAV.

The narration is synthesized *uncensored* (Kokoro speaks the real word); Step 4.5's forced
alignment then tells us exactly *when* each banned word is spoken. This module overlays a short
blur/beep SFX on top of those spans so the spoken word is masked, while the surrounding prosody and
total length are untouched. Two guarantees the user asked for:

  * **Loud (but not overpowering)** -- the SFX is peak-normalized then trimmed by `gain_db`, and the
    real word underneath is ducked to near-silence (`voice_duck_db`, default -30 dB), so the blur
    masks the word without being the only thing the viewer hears.
  * **A touch shorter than the word** -- the SFX (and the matching duck) span the aligned word
    interval *inset* by `sfx_inset_sec` on each edge, so the beep is slightly shorter than the word
    and doesn't drag. A word too short for the inset falls back to its exact interval (never left
    audible).

The WAV is rewritten in place (same path, same sample rate). The blur SFX is decoded once via the
bundled ffmpeg (so any container/codec works, e.g. the bundled blur_sfx.mp3) and cached resampled
to the narration's rate. numpy/soundfile (already pulled in by `[tts]`) do the mixing.
"""

from __future__ import annotations

from pathlib import Path

from ..video.background import _ffmpeg, _run

# A short linear fade (seconds) on each blur segment's edges, so tiling/trimming the SFX never
# leaves an audible click at the loop seam or the segment boundary.
_FADE_SEC = 0.008


class BlurCensor:
    """Overlays a blur SFX over banned-word spans in a narration WAV. The SFX is loaded lazily and
    cached resampled to the narration rate, so a whole run decodes it once."""

    def __init__(
        self,
        *,
        sfx_file: str,
        gain_db: float = 0.0,
        voice_duck_db: float = -30.0,
        sfx_inset_sec: float = 0.04,
        sample_rate: int = 24000,
    ):
        self.sfx_file = sfx_file
        self.gain_db = gain_db
        self.voice_duck_db = voice_duck_db
        self.sfx_inset_sec = max(0.0, sfx_inset_sec)
        self.sample_rate = sample_rate
        self._sfx = None  # peak-normalized mono float32 at `sample_rate`, decoded on first use

    def _load_sfx(self, sr: int):
        """Decode the SFX to mono float32 at `sr` via ffmpeg and peak-normalize it (cached)."""
        import numpy as np

        if self._sfx is not None:
            return self._sfx
        ffmpeg = _ffmpeg()
        # Decode straight to raw 32-bit float mono at the narration's rate -- no temp file.
        stdout, _ = _run(
            [ffmpeg, "-v", "error", "-i", str(self.sfx_file),
             "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
            capture=True,
        )
        a = np.frombuffer(stdout, dtype=np.float32).astype(np.float32).copy()
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        self._sfx = a / peak if peak > 0 else a
        return self._sfx

    def censor(self, wav_path: Path, intervals: list[tuple[float, float]]) -> int:
        """Overlay the blur SFX over each (start, end) span in `wav_path`, in place.

        Returns the number of spans blurred. Each span is inset by `sfx_inset_sec` on each edge (so
        the beep runs a touch shorter than the word; a word too short to inset uses its exact
        interval), ducks the underlying narration there by `voice_duck_db`, and adds the SFX
        (tiled/trimmed to that span) on top; the sum is clipped to [-1, 1]. Mono or stereo input is
        handled. A span list that's empty (no banned words) is a no-op.
        """
        import numpy as np
        import soundfile as sf

        if not intervals:
            return 0
        audio, sr = sf.read(str(wav_path), dtype="float32")
        sfx = self._load_sfx(sr)
        if sfx.size == 0:
            return 0
        n = audio.shape[0]
        level = 10 ** (self.gain_db / 20.0)
        duck = 10 ** (self.voice_duck_db / 20.0)
        fade = max(1, int(_FADE_SEC * sr))
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        count = 0
        for start, end in intervals:
            # Inset the span so the beep (and its duck) is a touch shorter than the word -- less
            # annoying than a beep that overruns it. If the word is too short to inset, fall back to
            # its exact interval so it's never left audible.
            s = max(0, int((start + self.sfx_inset_sec) * sr))
            e = min(n, int((end - self.sfx_inset_sec) * sr))
            if e <= s:
                s = max(0, int(start * sr))
                e = min(n, int(end * sr))
            if e <= s:
                continue
            seg_len = e - s
            # Tile (loop) the SFX to the span length and trim the tail. Re-peak-normalize this slice
            # so EVERY blur is uniformly loud -- a short word slices only the SFX's (often soft) onset,
            # which would otherwise be quiet. `level` (gain_db) then scales on top; the final clip guards.
            beep = np.resize(sfx, seg_len).astype(np.float32)
            seg_peak = float(np.max(np.abs(beep)))
            if seg_peak > 0:
                beep *= (0.97 / seg_peak)
            beep *= level
            f = min(fade, seg_len // 2)
            if f > 0:
                ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                beep[:f] *= ramp
                beep[-f:] *= ramp[::-1]
            # Duck the real word to near-silence, then add the (loud) blur over it.
            if channels == 1:
                audio[s:e] = audio[s:e] * duck + beep
            else:
                audio[s:e, :] = audio[s:e, :] * duck + beep[:, None]
            count += 1

        np.clip(audio, -1.0, 1.0, out=audio)
        sf.write(str(wav_path), audio, sr)
        return count
