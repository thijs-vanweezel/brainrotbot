from pathlib import Path

from brainrotbot.text.filter_words import filter_banned_words


def _write_map(tmp_path: Path) -> Path:
    p = tmp_path / "banned.toml"
    p.write_text(
        '[replacements]\nkill = "unalive"\ngun = "pew-pew"\n',
        encoding="utf-8",
    )
    return p


def test_basic_substitution_and_log(tmp_path):
    path = _write_map(tmp_path)
    out, reps = filter_banned_words("I will kill the boss with a gun.", path)
    assert "kill" not in out and "gun" not in out
    assert "unalive" in out and "pew-pew" in out
    by = {r.from_word: r.count for r in reps}
    assert by == {"kill": 1, "gun": 1}


def test_preserves_case(tmp_path):
    path = _write_map(tmp_path)
    out, _ = filter_banned_words("KILL. Kill. kill.", path)
    assert "UNALIVE" in out
    assert "Unalive" in out
    assert "unalive" in out


def test_whole_word_only(tmp_path):
    path = _write_map(tmp_path)
    # "skill" contains "kill" but must not be touched.
    out, reps = filter_banned_words("That is a great skill.", path)
    assert out == "That is a great skill."
    assert reps == []


def test_counts_multiple_occurrences(tmp_path):
    path = _write_map(tmp_path)
    out, reps = filter_banned_words("gun gun gun", path)
    assert out == "pew-pew pew-pew pew-pew"
    assert reps[0].count == 3
