from brainrotbot.text.clean import clean_text, html_to_text


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
    out = clean_text("", "I’d say “hi”—really…", prepend_title=False)
    assert "’" not in out and "“" not in out and "—" not in out
    assert "I'd say" in out
    assert '"hi"' in out
    assert "-really..." in out
