from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urldefrag, urlparse
import httpx


class _TextAndLinksParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self._in_style = False
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "script":
            self._in_script = True
            return
        if t == "style":
            self._in_style = True
            return
        if t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t == "script":
            self._in_script = False
        elif t == "style":
            self._in_style = False

    def handle_data(self, data):
        if self._in_script or self._in_style:
            return
        s = data.strip()
        if s:
            self.text_parts.append(s)


def html_to_text(html: str) -> str:
    parser = _TextAndLinksParser()
    parser.feed(html)
    text = "\n".join(parser.text_parts)
    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_links(html: str, base_url: str, *, limit: int = 50) -> list[str]:
    parser = _TextAndLinksParser()
    parser.feed(html)

    out: list[str] = []
    seen: set[str] = set()

    for href in parser.links:
        abs_url = urljoin(base_url, href)
        abs_url, _ = urldefrag(abs_url)

        p = urlparse(abs_url)
        if p.scheme not in ("http", "https"):
            continue

        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append(abs_url)

        if len(out) >= limit:
            break

    return out


def extract_title(html: str) -> str | None:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    t = re.sub(r"\s+", " ", m.group(1)).strip()
    return t or None


def fetch_url(url: str, *, timeout_s: float = 25.0, max_chars: int = 250_000) -> str:
    """
    Free fetch tool. Returns text, truncated.
    Safety:
      - only http(s)
      - follows redirects
      - truncates large responses
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Blocked: only http(s) URLs are allowed."

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "deep-research-agent/0.1"})
            r.raise_for_status()
            text = r.text or ""

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[TRUNCATED]\n"
        return text
    except Exception as e:
        return f"Fetch failed: {type(e).__name__}: {e}"