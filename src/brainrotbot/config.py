"""Configuration loading from settings.toml.

Reddit retrieval uses RSS feeds and needs no credentials. Paths in settings.toml are
resolved relative to the project root so the pipeline works from any directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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

    @property
    def video_opts(self) -> dict:
        return self.raw["video"]

    @property
    def edit_opts(self) -> dict:
        return self.raw.get("edit", {})

    @property
    def thumbnail_opts(self) -> dict:
        return self.raw.get("thumbnail", {})

    @property
    def thumbnail_font_file(self) -> str:
        """Absolute path to the title font, or "" if unset (resolved like the outro file)."""
        ff = self.thumbnail_opts.get("font_file", "")
        return str(self._resolve(ff)) if ff else ""

    @property
    def edit_outro_file(self) -> str:
        """Absolute path to the outro asset, or "" if unset (resolved like banned_words)."""
        of = self.edit_opts.get("outro_file", "")
        return str(self._resolve(of)) if of else ""

    @property
    def video_cookies_file(self) -> str:
        """Absolute path to the yt-dlp cookies file, or "" if unset.

        A relative path is resolved against the settings file's own directory (config/), so
        `cookies_file = "./yt_cookies.txt"` finds a cookies.txt sitting next to settings.toml.
        """
        cf = self.raw["video"].get("cookies_file", "")
        if not cf:
            return ""
        p = Path(cf)
        return str(p if p.is_absolute() else (self.settings_path.parent / p).resolve())

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
    def video_dir(self) -> Path:
        return self.data_dir / self.raw["paths"]["video_subdir"]

    @property
    def video_cache_dir(self) -> Path:
        return self.data_dir / self.raw["paths"]["video_cache_subdir"]

    @property
    def final_dir(self) -> Path:
        return self.data_dir / self.raw["paths"]["final_subdir"]

    @property
    def music_cache_dir(self) -> Path:
        # Step 5: NCS instrumental MP3s + scraped catalogue. Keyed by track UUID (stable across
        # runs), so cache hits accumulate naturally even with random per-story picks.
        return self.data_dir / self.raw["paths"].get("music_cache_subdir", "music_cache")

    @property
    def thumbnail_dir(self) -> Path:
        return self.data_dir / self.raw["paths"].get("thumbnail_subdir", "thumbnail")

    @property
    def thumbnail_cache_dir(self) -> Path:
        # Step 6: downloaded Pixabay backgrounds, keyed by image id (reused across runs when the
        # same image is picked again; otherwise a fresh random image per story).
        return self.data_dir / self.raw["paths"].get("thumbnail_cache_subdir", "thumbnail_cache")

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / self.raw["paths"]["ledger_file"]

    @property
    def banned_words_path(self) -> Path:
        return self._resolve(self.raw["paths"]["banned_words_file"])


def load_settings(path: Path | None = None) -> Settings:
    load_dotenv()  # pull secrets (e.g. PIXABAY_API_KEY) from a local .env into the environment
    path = path or DEFAULT_SETTINGS_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Settings(raw=raw, settings_path=path)
