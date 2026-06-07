"""Step 4.5: build burned-in word-by-word "brainrot" captions from the narration.

We already own the exact narration script (`entry.text["cleaned_body"]`, the same string Step 2
feeds Kokoro) and the clean TTS WAV, so this is a *timing* problem, not transcription. We use
**forced alignment** -- an MMS/Wav2Vec2 CTC model emits per-frame character probabilities, and a
Viterbi pass over the known character sequence yields per-word timestamps that always match the
script (no ASR drift). The words are grouped into short cues and written as an `.ass` subtitle file
with one Dialogue event per word interval (the active word "pops" in highlight_color); Step 4's
ffmpeg pass burns it in via the `subtitles=` filter.

Why a hand-rolled aligner instead of `ctc-forced-aligner`: that package ships a C++ extension that
won't build on Windows, and torchaudio's `forced_align` op would force a torch downgrade that breaks
the verified Kokoro env. transformers + torch (already present for TTS) + a ~40-line Viterbi (the
standard torchaudio-tutorial trellis) avoid both -- no compiled build, no ABI risk. Heavy deps
(transformers, torch, soundfile, soxr) are imported lazily so the core package stays importable
without the `[subtitles]` extra.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..text.censor import is_banned, mask_vowels

# HF model used by ctc-forced-aligner too: a 31-symbol MMS CTC aligner (blank=0, then lowercase
# latin + apostrophe). Emits ~50 frames/sec; we compute the exact sec/frame per clip.
_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
_SAMPLE_RATE = 16000
_BLANK_ID = 0
# Sentence-ending punctuation forces a cue break so a caption never straddles two sentences.
_SENT_END_RE = re.compile(r"[.!?]['\"]?$")


def _hex_to_ass(color: str) -> str:
    """`#RRGGBB` -> libass `&HBBGGRR&` (ASS stores colour byte-reversed, no alpha = opaque)."""
    h = color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{b}{g}{r}&".upper()


def _ass_time(seconds: float) -> str:
    """Seconds -> ASS timestamp `h:mm:ss.cc` (centiseconds)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _escape_text(text: str) -> str:
    """Neutralise ASS override syntax in display text (braces start overrides; \\N is a newline)."""
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


class SubtitleMaker:
    """Forced-aligns a narration to its script and renders word-by-word `.ass` captions.

    The CTC model + tokenizer load lazily on the first `make()` and are cached on the instance,
    so a whole run pays the load once (mirrors `KokoroSynthesizer`'s per-run pipeline cache).
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        language: str = "eng",  # recorded in meta; the MMS aligner is multilingual-by-romanization
        max_words_per_cue: int = 4,
        max_gap_sec: float = 0.6,
        font_name: str = "Anton",
        font_size: int = 90,
        primary_color: str = "#FFFFFF",
        highlight_color: str = "#FFE000",
        outline_color: str = "#000000",
        outline_width: int = 4,
        alignment: int = 5,
        margin_v: int = 120,
        uppercase: bool = True,
        scale_active: int = 110,
        width: int = 1080,
        height: int = 1920,
        banned_words: frozenset[str] = frozenset(),
    ):
        self.device = device
        # Words whose vowels are asterisked on screen ("fuck" -> "f*ck"); their spoken intervals are
        # also returned by make() so tts/censor.py can lay the blur SFX over them. Alignment itself
        # is unaffected -- it tokenizes the real (a-z) word, only the *displayed* glyphs change.
        self.banned_words = banned_words
        self.language = language
        self.max_words_per_cue = max_words_per_cue
        self.max_gap_sec = max_gap_sec
        self.font_name = font_name
        self.font_size = font_size
        self.primary_color = primary_color
        self.highlight_color = highlight_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.alignment = alignment
        self.margin_v = margin_v
        self.uppercase = uppercase
        self.scale_active = scale_active
        self.width, self.height = width, height
        self._model = None
        self._tokenizer = None
        self._vocab: dict[str, int] = {}
        self._resolved_device = None

    # --- model -------------------------------------------------------------
    def _load_model(self):
        """Get-or-build the CTC aligner model + tokenizer (lazy, heavy, optional)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCTC, AutoTokenizer

        if self.device in (None, "auto"):
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self.device
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        self._vocab = self._tokenizer.get_vocab()
        self._model = AutoModelForCTC.from_pretrained(_MODEL).eval().to(self._resolved_device)

    # --- alignment ---------------------------------------------------------
    def _emissions(self, wav16):
        """Run the CTC model -> log-prob emissions (T, C), as a CPU tensor for the Viterbi loop.

        Tries the resolved device first; on CUDA OOM (the 6 GB card is shared) it retries on CPU
        for this clip rather than aborting the step.
        """
        import torch

        x = torch.from_numpy(wav16).unsqueeze(0)
        for dev in (self._resolved_device, "cpu"):
            try:
                with torch.inference_mode():
                    logits = self._model.to(dev)(x.to(dev)).logits[0]
                return torch.log_softmax(logits.float().cpu(), dim=-1)
            except RuntimeError as exc:  # pragma: no cover -- OOM only on a real GPU
                if dev == "cpu" or "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
        raise RuntimeError("unreachable")

    def _tokenize(self, display_words: list[str]):
        """Flatten display words to in-vocab char token ids, tracking each token's word index.

        Only `[a-z']` survive (the aligner's symbol set); digits/punctuation/emoji map to nothing,
        so a word like "2023" contributes no anchor and gets interpolated timing later. This keeps
        alignment robust on ordinary prose without a full number-to-words normaliser.
        """
        tokens: list[int] = []
        tok_word: list[int] = []
        for wi, w in enumerate(display_words):
            for ch in re.sub(r"[^a-z']", "", w.lower()):
                tid = self._vocab.get(ch)
                if tid is not None:
                    tokens.append(tid)
                    tok_word.append(wi)
        return tokens, tok_word

    @staticmethod
    def _forced_align(emission, tokens):
        """Standard CTC forced-alignment trellis + backtrack (torchaudio-tutorial formulation).

        Returns, per target token, the (first_frame, last_frame) it occupies. Each frame is either
        the current token or blank; the path is monotonic so word order is preserved.
        """
        import torch

        T, N = emission.size(0), len(tokens)
        tok_t = torch.tensor(tokens)
        trellis = torch.full((T, N), -float("inf"))
        trellis[0, 0] = emission[0, tokens[0]]
        for t in range(1, T):
            trellis[t, 0] = trellis[t - 1, 0] + emission[t, _BLANK_ID]
            stay = trellis[t - 1, 1:] + emission[t, _BLANK_ID]
            adv = trellis[t - 1, :-1] + emission[t, tok_t[1:]]
            trellis[t, 1:] = torch.maximum(stay, adv)

        # Backtrack from the last token at the last frame.
        frames: dict[int, list[int]] = {}
        t, j = T - 1, N - 1
        while t >= 0:
            frames.setdefault(j, []).append(t)
            if j > 0:
                stay = trellis[t - 1, j] + emission[t, _BLANK_ID]
                adv = trellis[t - 1, j - 1] + emission[t, tokens[j]]
                if adv > stay:
                    j -= 1
            t -= 1
        return {j: (min(fs), max(fs)) for j, fs in frames.items()}

    def _align(self, audio_path: Path, text: str) -> list[dict]:
        """Forced-align `text` to `audio_path`; return [{text, start, end}] for every display word.

        Words with no in-vocab characters (numbers, emoji) carry no alignment anchor and have their
        timing interpolated from neighbours so every caption word still gets a sane on-screen span.
        """
        import numpy as np
        import soundfile as sf
        import soxr

        self._load_model()
        wav, sr = sf.read(str(audio_path))
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav16 = soxr.resample(wav.astype(np.float32), sr, _SAMPLE_RATE)
        audio_dur = len(wav16) / _SAMPLE_RATE

        display_words = text.split()
        if not display_words:
            return []
        tokens, tok_word = self._tokenize(display_words)
        spans = [None] * len(display_words)  # (start_sec, end_sec) per display word
        if tokens:
            emission = self._emissions(wav16)
            ratio = audio_dur / emission.size(0)  # sec per frame
            tok_frames = self._forced_align(emission, tokens)
            # Token frame-spans -> per-word frame-spans (first token's start .. last token's end).
            for j, (f0, f1) in tok_frames.items():
                wi = tok_word[j]
                lo, hi = (f0, f1) if spans[wi] is None else (
                    min(spans[wi][0], f0), max(spans[wi][1], f1))
                spans[wi] = (lo, hi)
            spans = [(s[0] * ratio, s[1] * ratio) if s is not None else None for s in spans]
        self._interpolate(spans, audio_dur)
        return [{"text": w, "start": s, "end": e}
                for w, (s, e) in zip(display_words, spans)]

    @staticmethod
    def _interpolate(spans: list, audio_dur: float) -> None:
        """Fill `None` (anchorless) word spans in place by evenly splitting the surrounding gap.

        Runs of missing words between two anchors share that interval; a leading run starts at 0,
        a trailing run ends at the audio length. If nothing aligned at all, spread evenly over the
        whole clip. Guarantees monotonic, non-overlapping, positive spans for the renderer.
        """
        n = len(spans)
        i = 0
        while i < n:
            if spans[i] is not None:
                i += 1
                continue
            j = i
            while j < n and spans[j] is None:
                j += 1
            lo = spans[i - 1][1] if i > 0 else 0.0
            hi = spans[j][0] if j < n else audio_dur
            if hi <= lo:
                hi = lo + 0.01 * (j - i)
            step = (hi - lo) / (j - i)
            for k in range(i, j):
                spans[k] = (lo + step * (k - i), lo + step * (k - i + 1))
            i = j

    # --- cue grouping + rendering -----------------------------------------
    def _group(self, words: list[dict]) -> list[list[dict]]:
        """Group words into short cues, breaking on size, sentence-end punctuation, or a long pause."""
        cues: list[list[dict]] = []
        cur: list[dict] = []
        for w in words:
            if cur and (
                len(cur) >= self.max_words_per_cue
                or w["start"] - cur[-1]["end"] > self.max_gap_sec
                or _SENT_END_RE.search(cur[-1]["text"])
            ):
                cues.append(cur)
                cur = []
            cur.append(w)
        if cur:
            cues.append(cur)
        return cues

    def _render_event(self, cue: list[dict], active: int) -> str:
        """One cue's text with word `active` highlighted (colour + optional scale-up)."""
        hi = _hex_to_ass(self.highlight_color)
        pri = _hex_to_ass(self.primary_color)
        s = self.scale_active
        out = []
        for i, w in enumerate(cue):
            txt = _escape_text(w["text"])
            if is_banned(txt, self.banned_words):
                txt = mask_vowels(txt)  # vowels -> '*' for the caption (audio is blurred separately)
            if self.uppercase:
                txt = txt.upper()
            if i == active:
                out.append(f"{{\\1c{hi}\\fscx{s}\\fscy{s}}}{txt}{{\\1c{pri}\\fscx100\\fscy100}}")
            else:
                out.append(txt)
        return " ".join(out)

    def _write_ass(self, cues: list[list[dict]], out_path: Path) -> None:
        """Write the `.ass`: header + V4+ style (Anton + outline) + one event per word interval.

        Within a cue each word's event runs from its own start to the next word's start (last word
        to its end), so the caption stays on screen continuously and the highlight steps word by
        word with no flicker gap between words.
        """
        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {self.width}\nPlayResY: {self.height}\n"
            "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{self.font_name},{self.font_size},{_hex_to_ass(self.primary_color)},"
            f"&H000000FF,{_hex_to_ass(self.outline_color)},&H64000000,-1,0,0,0,100,100,0,0,1,"
            f"{self.outline_width},0,{self.alignment},40,40,{self.margin_v},1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        lines = [header]
        for cue in cues:
            for i, w in enumerate(cue):
                start = w["start"]
                end = cue[i + 1]["start"] if i + 1 < len(cue) else w["end"]
                if end <= start:
                    end = start + 0.05
                lines.append(
                    f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
                    f"{self._render_event(cue, i)}\n"
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines), encoding="utf-8")

    def make(self, audio_path: Path, text: str, out_ass_path: Path) -> dict:
        """Align `text` to `audio_path` and write word-by-word captions to `out_ass_path`.

        Returns ledger meta. Raises on hard failures (missing audio, model load); the pipeline
        wraps the call so a failure just leaves the video uncaptioned.
        """
        words = self._align(Path(audio_path), text)
        cues = self._group(words)
        self._write_ass(cues, Path(out_ass_path))
        # Spoken intervals of the banned words, so tts/censor.py can lay the blur SFX over exactly
        # those spans (the blur length thus matches each word). Rounded for a tidy ledger record.
        banned_intervals = [
            [round(w["start"], 3), round(w["end"], 3)]
            for w in words if is_banned(w["text"], self.banned_words)
        ]
        return {
            "path": str(out_ass_path),
            "num_words": len(words),
            "num_cues": len(cues),
            "num_masked": len(banned_intervals),
            "banned_intervals": banned_intervals,
            "backend": "ctc-forced-align",
            "model": _MODEL,
            "language": self.language,
        }
