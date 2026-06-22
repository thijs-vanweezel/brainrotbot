"""Overlay a one-shot "ahem" SFX on the narration WAV right after the post title is read.

Classic brainrot pacing reads the post title as the hook, then a throat-clear before the story
body. The narration is one continuous WAV (title + body); Step 4.5's forced alignment tells us
exactly when the title's last word ends, and this module lays `resources/ahem.mp3` on top of the
narration starting at that instant. Unlike `tts/censor.py:BlurCensor` (which ducks the word it
masks), this is a *pure overlay* -- the narration keeps playing underneath and the total length is
unchanged (the SFX is truncated if it would run past the end), so the background clip (sized to
`assets.audio.duration_sec`) and the burned-in captions stay perfectly in sync.

The SFX is peak-normalized loud (~0.95) so it's clearly audible over the speech it briefly overlaps.
It's decoded once via the bundled ffmpeg (any container/codec works) and cached resampled to the
narration's rate. numpy/soundfile (already pulled in by `[tts]`) do the mixing.
"""

from __future__ import annotations

from pathlib import Path

from ..video.background import _ffmpeg, _run

# A short linear fade (seconds) on the SFX edges so the truncated tail never leaves an audible click.
_FADE_SEC = 0.008
# Peak level the SFX is normalized to before `gain_db` -- high so the ahem clearly cuts through the
# narration it overlaps (the final sum is clipped to [-1, 1]).
_SFX_PEAK = 0.95


class IntroSfx:
    """Overlays a one-shot SFX onto a narration WAV at a given timestamp. The SFX is loaded lazily
    and cached resampled to the narration rate, so a whole run decodes it once."""

    def __init__(self, *, sfx_file: str, gain_db: float = 0.0, offset_sec: float = -0.15,
                 sample_rate: int = 24000):
        self.sfx_file = sfx_file
        self.gain_db = gain_db
        self.offset_sec = offset_sec  # shift the start by this much (negative = a bit earlier)
        self.sample_rate = sample_rate
        self._sfx = None  # peak-normalized mono float32 at `sample_rate`, decoded on first use

    def _load_sfx(self, sr: int):
        """Decode the SFX to mono float32 at `sr` via ffmpeg and peak-normalize it (cached)."""
        import numpy as np

        if self._sfx is not None:
            return self._sfx
        # Decode straight to raw 32-bit float mono at the narration's rate -- no temp file.
        stdout, _ = _run(
            [_ffmpeg(), "-v", "error", "-i", str(self.sfx_file),
             "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
            capture=True,
        )
        a = np.frombuffer(stdout, dtype=np.float32).astype(np.float32).copy()
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        self._sfx = a * (_SFX_PEAK / peak) if peak > 0 else a
        return self._sfx

    def overlay(self, wav_path: Path, at_sec: float) -> bool:
        """Mix the SFX into `wav_path` in place, starting at `at_sec` + `offset_sec`. Returns True if
        applied.

        The narration is left playing underneath (no ducking); the SFX is added on top at
        `gain_db`, truncated so the WAV length is unchanged, with a short edge fade and a final clip
        to [-1, 1]. Mono or stereo input is handled. A start past the WAV end is a no-op.
        """
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32")
        sfx = self._load_sfx(sr)  # decoded/cached at the narration WAV's own rate
        if sfx.size == 0:
            return False
        n = audio.shape[0]
        s = max(0, int((at_sec + self.offset_sec) * sr))  # nudge earlier/later; never before 0
        if s >= n:
            return False  # nothing left to overlay onto

        seg = (sfx[: n - s] * (10 ** (self.gain_db / 20.0))).astype(np.float32)
        f = min(max(1, int(_FADE_SEC * sr)), seg.size // 2)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            seg[:f] *= ramp
            seg[-f:] *= ramp[::-1]
        e = s + seg.size
        if audio.ndim == 1:
            audio[s:e] += seg
        else:
            audio[s:e, :] += seg[:, None]

        np.clip(audio, -1.0, 1.0, out=audio)
        sf.write(str(wav_path), audio, sr)
        return True
