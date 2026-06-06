"""Data models for brainrotbot.

`Story` is the normalized representation of a Reddit post. `LedgerEntry` is the
durable, append-only record written for every selected story; its `assets`,
`upload`, and `metrics` sections are reserved now and populated by later pipeline
steps (TTS, editing, upload, analytics).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# Average narration pace used to estimate spoken duration (~150 words/min).
WORDS_PER_SECOND = 2.5


@dataclass
class Story:
    """A normalized Reddit self/text post.

    Sourced from RSS feeds, which expose the full text but not popularity numbers,
    so `score`/`upvote_ratio`/`num_comments`/`nsfw` may be None. `feed_rank` is the
    0-based position in the (already popularity-ordered) feed and stands in for score.
    """

    post_id: str
    subreddit: str
    title: str
    raw_body: str
    url: str
    author: str
    created_utc: float
    score: int | None = None
    upvote_ratio: float | None = None
    num_comments: int | None = None
    feed_rank: int | None = None
    flair: str | None = None
    nsfw: bool | None = None
    stickied: bool = False

    @property
    def word_count(self) -> int:
        return len(self.raw_body.split())


@dataclass
class Replacement:
    """A single banned-word substitution made during cleaning."""

    from_word: str
    to_word: str
    count: int


@dataclass
class LedgerEntry:
    """Durable record for one selected story. Written as one JSONL line."""

    id: str
    created_at: float
    status: str
    source: dict[str, Any]
    text: dict[str, Any]
    content_analysis: dict[str, Any] = field(default_factory=dict)
    assets: dict[str, Any] = field(default_factory=dict)
    upload: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        return cls(**data)

    @classmethod
    def from_story(
        cls,
        story: Story,
        cleaned_body: str,
        replacements: list[Replacement],
    ) -> "LedgerEntry":
        """Build a fresh ledger entry from a story and its cleaned text."""
        word_count = len(cleaned_body.split())
        return cls(
            id=uuid.uuid4().hex,
            created_at=time.time(),
            status="cleaned",
            source={
                "platform": "reddit",
                "subreddit": story.subreddit,
                "post_id": story.post_id,
                "url": story.url,
                "author": story.author,
                "feed_rank": story.feed_rank,
                "score": story.score,
                "upvote_ratio": story.upvote_ratio,
                "num_comments": story.num_comments,
                "flair": story.flair,
                "created_utc": story.created_utc,
                "nsfw": story.nsfw,
            },
            text={
                "title": story.title,
                "raw_body": story.raw_body,
                "cleaned_body": cleaned_body,
                "word_count": word_count,
                "est_speech_seconds": round(word_count / WORDS_PER_SECOND, 1),
                "banned_words_replaced": [asdict(r) for r in replacements],
            },
            content_analysis={"topic": None, "style": None, "tone": None, "hook_type": None},
            assets={"audio_path": None, "background_video": None, "final_video": None, "music_path": None},
            upload={"tiktok_id": None, "posted_at": None, "caption": None, "hashtags": None},
            metrics={"views": None, "likes": None, "comments": None, "shares": None, "retention": None},
        )
