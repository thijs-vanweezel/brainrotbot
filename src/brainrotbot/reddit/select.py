"""Popularity-based eligibility filtering and ranking of candidate stories."""

from __future__ import annotations

from collections.abc import Iterable

from ..models import Story

# Markers Reddit leaves in the body when a post is pulled.
_REMOVED_MARKERS = {"[removed]", "[deleted]"}


def is_eligible(story: Story, selection: dict, *, seen_ids: set[str]) -> bool:
    if story.post_id in seen_ids:
        return False
    if story.stickied:
        return False
    if not story.raw_body.strip() or story.raw_body.strip().lower() in _REMOVED_MARKERS:
        return False
    # NSFW flag is unavailable via RSS (None) -- only filter when known.
    if story.nsfw and not selection.get("allow_nsfw", False):
        return False
    # Score is unavailable via RSS (None) -- only apply min_score when known.
    if story.score is not None and story.score < int(selection.get("min_score", 0)):
        return False
    words = story.word_count
    if words < int(selection.get("min_words", 0)):
        return False
    if words > int(selection.get("max_words", 10**9)):
        return False
    return True


def rank_key(story: Story) -> tuple[int, float]:
    """Sort key (descending) ranking better stories first.

    With real numbers, rank by upvotes weighted by agreement plus engagement. Without
    them (RSS), rank by feed position -- the feed is already popularity-ordered, so a
    lower `feed_rank` is better, negated here so descending sort keeps feed order.
    """
    if story.score is not None:
        engagement = 0.5 * (story.num_comments or 0)
        return (1, story.score * max(story.upvote_ratio or 0.01, 0.01) + engagement)
    return (0, -(story.feed_rank if story.feed_rank is not None else 10**9))


def select_stories(
    stories: Iterable[Story],
    selection: dict,
    *,
    seen_ids: set[str] | None = None,
) -> list[Story]:
    seen = seen_ids or set()
    eligible = [s for s in stories if is_eligible(s, selection, seen_ids=seen)]
    eligible.sort(key=rank_key, reverse=True)
    top_k = int(selection.get("top_k", 5))
    return eligible[:top_k]
