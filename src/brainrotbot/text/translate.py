"""Step 1.5: translate the cleaned English story into the target languages before TTS.

We fan a story out into several language variants (original English + French/Spanish/Chinese);
each variant is narrated, edited and uploaded as its own post so Step 8 can A/B which language
performs. Translation is local and offline: Meta's NLLB-200-distilled-600M run through
CTranslate2 with int8 quantization -- one model covers every target language, it's deterministic
(unlike an LLM, no chatter to strip), and int8 keeps it fast on CPU / a 6 GB GPU.

The model is converted to a quantized CTranslate2 directory once (lazily, on first use) and cached
on disk; the HF tokenizer provides NLLB's SentencePiece encoding + the per-language token prefix.

`Translator` mirrors `tts.synthesize.KokoroSynthesizer`: a lazily-built, reused-across-the-run
holder whose `translate()` returns the localized text. English is a passthrough (no model load).
"""

from __future__ import annotations

import re
from pathlib import Path

# Per-language settings. `nllb` is the FLORES-200 code CTranslate2 needs as the target prefix;
# `kokoro` is the Kokoro lang_code the TTS step uses to pick the matching voice pool. English is
# the source language (no translation) and maps to Kokoro's American-English voices.
SOURCE_NLLB = "eng_Latn"
LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"nllb": "",         "kokoro": "a", "label": "English"},
    "fr": {"nllb": "fra_Latn", "kokoro": "f", "label": "French"},
    "es": {"nllb": "spa_Latn", "kokoro": "e", "label": "Spanish"},
    "zh": {"nllb": "zho_Hans", "kokoro": "z", "label": "Chinese"},
}

# Chinese is written without inter-word spaces; everything else joins sentences with a space.
_NO_SPACE_LANGS = {"zh"}

# Split English source into sentences so each stays well under NLLB's ~512-token limit. Keeps the
# trailing punctuation with the sentence; also breaks on blank lines (paragraph boundaries).
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*\s*", re.UNICODE)


def kokoro_lang_code(lang: str) -> str:
    """Kokoro lang_code for a translation language key (defaults to American English)."""
    return LANGUAGES.get(lang, LANGUAGES["en"])["kokoro"]


def _split_sentences(text: str) -> list[str]:
    parts = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


class Translator:
    """Lazily-built NLLB/CTranslate2 translator, reused across a whole run.

    The quantized CT2 model is converted on first non-English use and cached under `cache_dir`;
    the tokenizer is loaded alongside it. Subsequent calls reuse both. `model_name` is the HF id
    converted into the cache.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str | None = "auto",
        compute_type: str = "int8",
    ):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.compute_type = compute_type
        self._device = device
        self._translator = None  # ctranslate2.Translator, built on first use
        self._tokenizer = None   # transformers tokenizer, built on first use
        self.model_label = f"nllb-200-distilled-600M/{compute_type}"

    # -- lazy setup -----------------------------------------------------------
    def _ct2_dir(self) -> Path:
        """Directory holding the converted, quantized CT2 model (a marker file confirms success)."""
        return self.cache_dir / (self.model_name.replace("/", "__") + f".{self.compute_type}")

    def _resolve_device(self) -> str:
        """Pick the CTranslate2 device. `auto` -> CPU: CT2's CUDA needs its own cuBLAS/cuDNN libs
        (not torch's) which aren't always present, and VRAM is shared with Kokoro/MMS -- int8
        NLLB-600M is fast on CPU. `cuda` is honored only if CT2 actually sees a CUDA device."""
        want = (self._device or "auto").lower()
        if want != "cuda":
            return "cpu"
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        import ctranslate2
        from transformers import AutoTokenizer

        ct2_dir = self._ct2_dir()
        if not (ct2_dir / "model.bin").is_file():
            # One-time conversion: download the HF model and write a quantized CT2 copy to disk.
            from ctranslate2.converters import TransformersConverter
            print(f"[brainrotbot] Converting {self.model_name} -> {self.compute_type} CT2 "
                  f"(one-time, this can take a few minutes) ...")
            ct2_dir.parent.mkdir(parents=True, exist_ok=True)
            TransformersConverter(self.model_name).convert(
                str(ct2_dir), quantization=self.compute_type, force=True
            )
        device = self._resolve_device()
        self._translator = ctranslate2.Translator(str(ct2_dir), device=device, compute_type=self.compute_type)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=SOURCE_NLLB)

    # -- translation ----------------------------------------------------------
    def translate(self, text: str, lang: str) -> str:
        """Translate English `text` into `lang` (a LANGUAGES key). English/unknown -> unchanged."""
        meta = LANGUAGES.get(lang)
        if not text or not meta or not meta["nllb"]:
            return text  # English passthrough -- no model load
        self._ensure_loaded()
        tok, translator = self._tokenizer, self._translator

        sentences = _split_sentences(text) or [text]
        sources = [tok.convert_ids_to_tokens(tok.encode(s)) for s in sentences]
        target_prefix = [[meta["nllb"]]] * len(sources)
        results = translator.translate_batch(
            sources, target_prefix=target_prefix, beam_size=2, max_batch_size=16
        )
        out_sentences = []
        for res in results:
            # hypotheses[0] starts with the target-language token we supplied -- drop it. Filter the
            # rare <unk> NLLB emits for an unmappable char so it's never spoken/printed literally.
            hyp = [t for t in res.hypotheses[0][1:] if t != "<unk>"]
            out_sentences.append(tok.decode(tok.convert_tokens_to_ids(hyp)))
        joiner = "" if lang in _NO_SPACE_LANGS else " "
        return joiner.join(s.strip() for s in out_sentences if s.strip())
