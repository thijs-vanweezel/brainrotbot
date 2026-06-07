"""Step 2: narrate a cleaned story to a WAV file with Kokoro-82M.

Kokoro is a small (82M), Apache-2.0, multilingual TTS model. Each language needs its
own `KPipeline` (and matching voices), so the synthesizer caches one pipeline per
`lang_code` and builds them lazily -- pipeline construction is the only slow part, so
reusing it across every story in a run keeps the pipeline fast. Output is 24 kHz mono WAV
(lossless, no encoder needed; the later editing step muxes it onto the background video).

All heavy deps (kokoro, numpy, soundfile) are imported lazily inside the methods so the
core package stays importable -- and the test suite stays runnable -- without the `[tts]`
extra installed.
"""

from __future__ import annotations

import random
from pathlib import Path

# Kokoro always synthesizes at 24 kHz.
KOKORO_SAMPLE_RATE = 24000


def pick_voice(voices: list[str], rng: random.Random | None = None) -> str:
    """Random voice from a language's pool (one per story; rng injectable for tests).

    A uniform per-story pick (not a deterministic rotation) so every voice keeps getting
    exercised regardless of run length; the chosen voice is logged per story to feed the
    Step 6 A/B analytics on what performs best.
    """
    if not voices:
        raise ValueError("voice pool is empty")
    return (rng or random).choice(voices)


class KokoroSynthesizer:
    """Lazily-built, per-language cache of Kokoro pipelines, reused across a run."""

    def __init__(self, *, device: str | None = "auto", sample_rate: int = KOKORO_SAMPLE_RATE):
        # device "auto" -> let Kokoro pick CUDA/CPU; otherwise honour the explicit choice.
        self._device = None if device in (None, "auto") else device
        self.sample_rate = sample_rate
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, lang_code: str):
        """Get-or-build the KPipeline for one language."""
        if lang_code not in self._pipelines:
            from kokoro import KPipeline  # heavy + optional; imported on first use

            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, device=self._device)
        return self._pipelines[lang_code]

    def synthesize(
        self,
        text: str,
        out_path: Path,
        *,
        voice: str,
        lang_code: str,
        speed: float = 1.0,
    ) -> dict:
        """Narrate `text` to `out_path` (WAV); return audio metadata for the ledger."""
        import numpy as np
        import soundfile as sf

        pipeline = self._pipeline(lang_code)

        # Kokoro yields one result per chunk (newer: `.audio`; older: a 3-tuple). Each
        # chunk's audio is a torch tensor or array; coerce to float32 numpy and concat.
        chunks = []
        for result in pipeline(text, voice=voice, speed=speed):
            audio = result.audio if hasattr(result, "audio") else result[-1]
            if hasattr(audio, "detach"):  # torch.Tensor
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro produced no audio (empty text?)")
        audio = np.concatenate(chunks)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, self.sample_rate)

        return {
            "audio_path": str(out_path),
            "voice": voice,
            "lang_code": lang_code,
            "duration_sec": round(len(audio) / self.sample_rate, 2),
            "sample_rate": self.sample_rate,
        }
