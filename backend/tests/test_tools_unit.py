from deep_research_agent.tools import html_to_text, extract_links, extract_title


def test_extract_title():
    html = "<html><head><title> Hello   World </title></head><body>x</body></html>"
    assert extract_title(html) == "Hello World"


def test_html_to_text_strips_script_style():
    html = """
    <html>
      <head>
        <style>.x{color:red}</style>
        <script>alert(1)</script>
        <title>T</title>
      </head>
      <body>
        <h1>Hello</h1>
        <p>World</p>
      </body>
    </html>
    """
    text = html_to_text(html)
    assert "alert" not in text
    assert "color:red" not in text
    assert "Hello" in text
    assert "World" in text


def test_extract_links_normalizes_and_filters():
    html = """
    <a href="/a">A</a>
    <a href="https://example.com/b#frag">B</a>
    <a href="mailto:test@example.com">M</a>
    """
    links = extract_links(html, "https://example.com/base", limit=10)
    assert "https://example.com/a" in links
    assert "https://example.com/b" in links
    assert all(not l.startswith("mailto:") for l in links)