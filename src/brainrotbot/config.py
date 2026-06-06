"""Configuration loading from settings.toml.

Reddit retrieval uses RSS feeds and needs no credentials. Paths in settings.toml are
resolved relative to the project root so the pipeline works from any directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# config.py lives at src/brainrotbot/config.py -> root is three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.toml"


@dataclass
class Settings:
    raw: dict
    settings_path: Path

    # --- convenience accessors -------------------------------------------------
    @property
    def subreddits(self) -> list[str]:
        return list(self.raw["reddit"]["subreddits"])

    @property
    def time_filter(self) -> str:
        return self.raw["reddit"]["time_filter"]

    @property
    def limit_per_sub(self) -> int:
        return int(self.raw["reddit"]["limit_per_sub"])

    @property
    def selection(self) -> dict:
        return self.raw["selection"]

    @property
    def text_opts(self) -> dict:
        return self.raw["text"]

    @property
    def tts_opts(self) -> dict:
        return self.raw["tts"]

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["data_dir"])

    @property
    def stories_dir(self) -> Path:
        return self.data_dir / self.raw["paths"]["stories_subdir"]

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / self.raw["paths"]["audio_subdir"]

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / self.raw["paths"]["ledger_file"]

    @property
    def banned_words_path(self) -> Path:
        return self._resolve(self.raw["paths"]["banned_words_file"])


def load_settings(path: Path | None = None) -> Settings:
    path = path or DEFAULT_SETTINGS_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Settings(raw=raw, settings_path=path)
