from brainrotbot.text.clean import clean_text, expand_reddit_abbreviations, html_to_text


def test_strips_markdown_links_and_urls():
    body = "Check [this](https://example.com) and https://foo.bar now."
    out = clean_text("", body, prepend_title=False)
    assert "https://" not in out
    assert "this" in out
    assert "[" not in out and "]" not in out


def test_strips_emphasis_and_headings():
    body = "# Title\n\nThis is **bold** and *italic* and `code`."
    out = clean_text("", body, prepend_title=False)
    assert "**" not in out and "*" not in out and "`" not in out
    assert "#" not in out
    assert "bold" in out and "italic" in out


def test_strips_edit_and_tldr_lines():
    body = "The story happened.\n\nEdit: thanks for the gold!\nTL;DR: it was wild."
    out = clean_text("", body, prepend_title=False, strip_edits=True, strip_tldr=True)
    assert "thanks for the gold" not in out
    assert "it was wild" not in out
    assert "The story happened." in out


def test_strips_reddit_rss_footer():
    """Reddit RSS tails every self-post with 'submitted by /u/<name>' + '[link] [comments]'.
    Both lines must be dropped or Kokoro narrates them at the end of the video."""
    # Verbatim from data/stories/1twj8pj.json: one leading space, then both lines.
    body = (
        "The actual story body ends here.\n\n"
        " submitted by /u/Substantial-Ad4756\n"
        " [link] [comments]"
    )
    out = clean_text("", body, prepend_title=False)
    assert "submitted by" not in out.lower()
    assert "/u/" not in out
    assert "[link]" not in out
    assert "[comments]" not in out
    assert out.endswith("The actual story body ends here.")


def test_does_not_eat_bare_bracket_link_in_prose():
    """A standalone '[link]' reference mid-paragraph must survive -- only the
    footer pattern '[link] [comments]' gets stripped."""
    body = "I clicked the [link] and nothing happened."
    out = clean_text("", body, prepend_title=False)
    assert "[link]" in out


# --- Reddit abbreviation expansion (so Kokoro narrates words, not letters) ------

def test_expands_aita_family_of_acronyms():
    """Sub-verdict acronyms must expand to prose. AITAH (5 chars) wins over AITA (4)."""
    assert expand_reddit_abbreviations("AITA for asking?") == "Am I the asshole for asking?"
    assert expand_reddit_abbreviations("AITAH?") == "Am I the asshole here?"
    assert expand_reddit_abbreviations("Verdict: NTA, but ESH a bit") == \
        "Verdict: Not the asshole, but Everyone sucks here a bit"


def test_expands_relationship_shorthand():
    out = expand_reddit_abbreviations("My BF told my MIL that his SIL is lying.")
    assert "boyfriend" in out and "mother-in-law" in out and "sister-in-law" in out
    assert "BF" not in out and "MIL" not in out and "SIL" not in out


def test_abbreviation_expansion_respects_word_boundaries():
    """'drop' contains 'OP' but must not expand; lowercase 'so'/'bf' are real English."""
    out = expand_reddit_abbreviations("I dropped the ball; so it's bf time.")
    # OP inside 'drop' / 'dropped' is not a standalone word -> untouched.
    assert "dropped" in out
    # Lowercase tokens are not Reddit acronyms -> untouched.
    assert " so " in out
    assert " bf " in out


def test_expands_age_gender_markers_in_parens():
    """(39F) -> '39 year old female'; (M28) -> '28 year old male'. Outside parens we don't
    expand because 'F35 hours' would be a false positive."""
    out = expand_reddit_abbreviations("I (39F) and my BF (38M) went riding.")
    assert "39 year old female" in out
    assert "38 year old male" in out
    assert "(F39)" not in out and "(38M)" not in out
    # And the leading-letter form:
    assert "32 year old female" in expand_reddit_abbreviations("(F32)")
    # Bare "F35" should NOT be expanded -- only the parens form is safe.
    assert "F35" in expand_reddit_abbreviations("I worked F35 hours.")


def test_clean_text_expands_in_title_and_body():
    """clean_text() applies the expansion to the assembled hook+body in one pass."""
    out = clean_text("AITA for ignoring my MIL?", "My BF said it was fine.", prepend_title=True)
    assert out.startswith("Am I the asshole for ignoring my mother-in-law?")
    assert "boyfriend" in out
    assert "AITA" not in out and "MIL" not in out and "BF" not in out


def test_prepends_title_as_hook():
    out = clean_text("My crazy day", "Then this happened.", prepend_title=True)
    assert out.startswith("My crazy day")
    assert "Then this happened." in out


def test_collapses_whitespace_and_blank_lines():
    body = "line one    with   spaces\n\n\n\nline two"
    out = clean_text("", body, prepend_title=False)
    assert "    " not in out
    assert "\n\n\n" not in out


# --- html_to_text (RSS self-text arrives as rendered HTML) ---------------------

def test_html_to_text_strips_tags():
    out = html_to_text("<p>Hello <strong>world</strong></p>")
    assert "<" not in out and ">" not in out
    assert "Hello world" in out


def test_html_to_text_unescapes_entities():
    out = html_to_text("<p>Tom &amp; Jerry said &quot;hi&quot; &gt;_&lt;</p>")
    assert "&amp;" not in out and "&quot;" not in out
    assert "Tom & Jerry" in out
    assert '"hi"' in out


def test_html_to_text_paragraphs_and_br_become_newlines():
    out = html_to_text("<p>one</p><p>two</p>three<br>four")
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert lines == ["one", "two", "three", "four"]


def test_html_to_text_empty():
    assert html_to_text("") == ""
    assert html_to_text(None) == ""


def test_normalizes_smart_punctuation_for_tts():
    # \uXXXX escapes avoid embedded curly-quote chars in the source (original
    # test used curly quotes AS delimiters, causing SyntaxError on Python 3.12).
    body = "\u201cI\u2019d say \u201chi\u201d\u2014really\u2026\u201d"
    out = clean_text("", body, prepend_title=False)
    assert "\u2018" not in out and "\u201c" not in out  # curly quotes gone
    assert "\u2014" not in out and "\u2013" not in out  # em/en dashes gone
    assert "I'd say" in out   # right-single-quote normalized to straight apostrophe
    assert '"hi"' in out      # curly double quotes normalized to straight
    assert "-really..." in out  # em-dash -> hyphen, ellipsis -> "..."

def test_title_asterisks_are_stripped():
    """Asterisks in the title (markdown emphasis or self-censored words) must be stripped
    before they reach Kokoro; otherwise TTS verbalises them as 'asterisk'. Text survives."""
    out = clean_text(
        "AITA for telling my *sister* to f*ck off?",
        "The whole story.",
        prepend_title=True,
    )
    assert "*" not in out, f"stray asterisk survived: {out!r}"
    assert "sister" in out
    # The word-core survives (vowel masking happens in text/censor.py, not clean_text).
    # "_strip_inline_markup" removes '*', leaving "fck" (the vowel was the asterisk).
    assert "fck" in out or "fuck" in out
