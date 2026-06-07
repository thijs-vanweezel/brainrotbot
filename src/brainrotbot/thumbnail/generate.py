"""Step 6: build a 9:16 thumbnail PNG -- a Pixabay background with the post title overlaid.

Per story: pick a random (category, term) from the curated pool, fetch Pixabay hits, pick one,
download+cache it, cover-crop to the TikTok frame, then overlay the title in bold white text
(dark stroke + a soft scrim band for legibility over any image). Output: data/thumbnail/<id>.png.

Mirrors video/background.py:BackgroundVideoMaker -- a config-holding maker whose `make()`
returns ledger metadata (search category/term, image provenance) and whose downloads are cached
on disk. All the heavy lifting is Pillow; the network/pick helpers live in images.py.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .images import download_image, pick_image, pick_term, search_images

# System fallbacks tried (in order) when no font_file is configured / it's missing.
_FALLBACK_FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]

# Translated (Step 1.5) Chinese/Japanese titles need a CJK-capable font -- the brainrot Anton font
# is latin-only and would render Han glyphs as empty boxes. Detect CJK and swap to a system CJK font.
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")
_CJK_FONTS = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _load_font(font_file: str, size: int) -> ImageFont.FreeTypeFont:
    """Configured TTF -> Arial Bold -> Arial -> Pillow's built-in font (last resort)."""
    for cand in [font_file, *_FALLBACK_FONTS]:
        if cand and Path(cand).is_file():
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)  # Pillow >=10.1 honours size
    except TypeError:
        return ImageFont.load_default()


def _cover_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to fill w x h then center-crop (same 9:16 fill as the video crop, in PIL)."""
    scale = max(w / img.width, h / img.height)
    nw, nh = round(img.width * scale), round(img.height * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


class ThumbnailMaker:
    """Turns the curated search-term pool + a title into per-story 9:16 thumbnail PNGs."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        search_terms: dict[str, list[str]],
        width: int = 1080,
        height: int = 1920,
        font_file: str = "",
        font_max_size: int = 110,
        font_min_size: int = 48,
        max_lines: int = 5,
        margin: int = 80,
        text_color: str = "white",
        stroke_color: str = "black",
        stroke_width: int = 6,
        scrim_opacity: float = 0.35,
        api_key: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.search_terms = search_terms
        self.width, self.height = width, height
        self.font_file = font_file
        self.font_max_size, self.font_min_size = font_max_size, font_min_size
        self.max_lines, self.margin = max_lines, margin
        self.text_color, self.stroke_color, self.stroke_width = text_color, stroke_color, stroke_width
        self.scrim_opacity = scrim_opacity
        self.api_key = api_key

    def _font_for_title(self, title: str) -> str:
        """Configured font normally; a system CJK font for Chinese/Japanese (Anton lacks Han glyphs)."""
        if _has_cjk(title):
            for cand in _CJK_FONTS:
                if Path(cand).is_file():
                    return cand
        return self.font_file

    def _fit_text(self, draw: ImageDraw.ImageDraw, title: str, font_file: str):
        """Wrap+auto-shrink the title to fit the text box; return (font, lines, line_height).

        CJK text has no spaces, so it wraps per character; latin text wraps per word.
        """
        is_cjk = _has_cjk(title)
        units = list(title) if is_cjk else title.split()
        sep = "" if is_cjk else " "
        max_w = self.width - 2 * self.margin
        max_h = int(self.height * 0.5)
        best = None
        for size in range(self.font_max_size, self.font_min_size - 1, -4):
            font = _load_font(font_file, size)
            lines, cur = [], ""
            for unit in units:
                trial = f"{cur}{sep}{unit}" if cur else unit
                tw = draw.textbbox((0, 0), trial, font=font, stroke_width=self.stroke_width)[2]
                if tw <= max_w or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = unit
            if cur:
                lines.append(cur)
            # Per-line height from the font metrics (stable across lines), + 20% leading.
            asc, desc = font.getmetrics()
            line_h = int((asc + desc + 2 * self.stroke_width) * 1.2)
            total_h = line_h * len(lines)
            best = (font, lines, line_h)
            if len(lines) <= self.max_lines and total_h <= max_h:
                return best
        return best  # smallest size reached -- use it even if slightly overflowing

    def pick(self) -> dict:
        """Choose + download one background image; return its provenance (no title drawn yet).

        Split out from `render` so the language fan-out (Step 1.5) can pick ONE image per base story
        and reuse it across every language variant (controlled A/B + a single Pixabay call). Tries one
        alternate search term if the first yields no usable hits, then raises.
        """
        category, term = pick_term(self.search_terms)
        hits = search_images(term, self.api_key, min_width=self.width, min_height=self.height)
        if not hits:
            category, term = pick_term(self.search_terms)
            hits = search_images(term, self.api_key, min_width=self.width, min_height=self.height)
        if not hits:
            raise RuntimeError(f"no Pixabay images for term '{term}' (check PIXABAY_API_KEY)")
        hit = pick_image(hits)
        img_path = download_image(hit, self.cache_dir)
        return {"category": category, "term": term, "hit": hit, "img_path": str(img_path)}

    def make(self, title: str, out_path: Path) -> dict:
        """Build the thumbnail for `title` at `out_path` (pick a fresh image, then render)."""
        return self.render(title, out_path, self.pick())

    def render(self, title: str, out_path: Path, picked: dict) -> dict:
        """Overlay `title` onto a previously `pick()`ed background; return ledger metadata."""
        category, term, hit = picked["category"], picked["term"], picked["hit"]
        font_file = self._font_for_title(title)

        base = _cover_crop(Image.open(picked["img_path"]).convert("RGB"), self.width, self.height).convert("RGBA")
        draw = ImageDraw.Draw(base)
        font, lines, line_h = self._fit_text(draw, title, font_file)

        block_h = line_h * len(lines)
        top = int(self.height * 0.30)                 # title block sits in the upper third
        # Semi-transparent dark band behind the text for legibility over any background.
        if self.scrim_opacity > 0:
            scrim = Image.new("RGBA", base.size, (0, 0, 0, 0))
            pad = self.margin // 2
            ImageDraw.Draw(scrim).rectangle(
                [0, top - pad, self.width, top + block_h + pad],
                fill=(0, 0, 0, int(255 * self.scrim_opacity)),
            )
            base = Image.alpha_composite(base, scrim)
            draw = ImageDraw.Draw(base)

        y = top
        for line in lines:
            w = draw.textbbox((0, 0), line, font=font, stroke_width=self.stroke_width)[2]
            draw.text(
                ((self.width - w) // 2, y), line, font=font,
                fill=self.text_color, stroke_width=self.stroke_width, stroke_fill=self.stroke_color,
            )
            y += line_h

        out_path.parent.mkdir(parents=True, exist_ok=True)
        base.convert("RGB").save(out_path)
        return {
            "path": str(out_path),
            "width": self.width,
            "height": self.height,
            "search_category": category,
            "search_term": term,
            "image_id": hit.id,
            "image_page_url": hit.page_url,
            "image_url": hit.image_url,
            "title_rendered": title,
        }
