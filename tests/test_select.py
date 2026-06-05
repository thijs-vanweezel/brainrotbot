from brainrotbot.models import Story
from brainrotbot.reddit.select import is_eligible, rank_key, select_stories

SELECTION = {
    "min_score": 100,
    "allow_nsfw": False,
    "min_words": 5,
    "max_words": 50,
    "top_k": 2,
}


def make_story(**kw) -> Story:
    """Default: an RSS-style story (no score, has feed_rank)."""
    base = dict(
        post_id="abc",
        subreddit="stories",
        title="t",
        raw_body="word " * 10,  # 10 words
        url="u",
        author="a",
        created_utc=0.0,
        feed_rank=0,
    )
    base.update(kw)
    return Story(**base)


def test_eligible_happy_path():
    assert is_eligible(make_story(), SELECTION, seen_ids=set())


def test_min_score_ignored_when_score_absent():
    # RSS stories have score=None -> min_score must not reject them.
    assert is_eligible(make_story(score=None), SELECTION, seen_ids=set())


def test_min_score_applied_when_score_present():
    assert not is_eligible(make_story(score=50), SELECTION, seen_ids=set())
    assert is_eligible(make_story(score=500), SELECTION, seen_ids=set())


def test_rejects_too_short_and_too_long():
    assert not is_eligible(make_story(raw_body="one two"), SELECTION, seen_ids=set())
    assert not is_eligible(make_story(raw_body="w " * 100), SELECTION, seen_ids=set())


def test_rejects_nsfw_when_disallowed():
    assert not is_eligible(make_story(nsfw=True), SELECTION, seen_ids=set())


def test_rejects_removed_and_stickied():
    assert not is_eligible(make_story(raw_body="[removed]"), SELECTION, seen_ids=set())
    assert not is_eligible(make_story(stickied=True), SELECTION, seen_ids=set())


def test_dedup_against_seen_ids():
    assert not is_eligible(make_story(post_id="dup"), SELECTION, seen_ids={"dup"})


def test_select_ranks_by_feed_order_when_no_score():
    stories = [
        make_story(post_id="third", feed_rank=2),
        make_story(post_id="first", feed_rank=0),
        make_story(post_id="second", feed_rank=1),
    ]
    out = select_stories(stories, SELECTION, seen_ids=set())
    assert [s.post_id for s in out] == ["first", "second"]


def test_select_ranks_by_score_when_present():
    stories = [
        make_story(post_id="low", score=200, upvote_ratio=0.9, num_comments=0),
        make_story(post_id="high", score=900, upvote_ratio=0.9, num_comments=100),
        make_story(post_id="mid", score=500, upvote_ratio=0.9, num_comments=10),
    ]
    out = select_stories(stories, SELECTION, seen_ids=set())
    assert [s.post_id for s in out] == ["high", "mid"]


def test_rank_key_feed_rank_lower_is_better():
    top = make_story(feed_rank=0)
    lower = make_story(feed_rank=5)
    assert rank_key(top) > rank_key(lower)
