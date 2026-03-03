from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote, urljoin, urldefrag, urlparse

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


def _is_blocked_host(host: str) -> bool:
    h = host.strip().lower()
    if not h:
        return True
    if h in {"localhost"}:
        return True
    if h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except Exception:
        return False


def _validate_url(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("only http(s) urls allowed")
    if p.username or p.password:
        raise ValueError("userinfo not allowed")
    if not p.hostname:
        raise ValueError("missing host")
    if _is_blocked_host(p.hostname):
        raise ValueError("blocked host")
    return url


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _jina_reader_url(url: str) -> str:
    encoded = quote(url, safe=":/?&=%#@+;,")
    return f"https://r.jina.ai/{encoded}"


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    url: str
    final_url: str
    status_code: int
    content_type: str
    extracted_text: str
    title: Optional[str]
    truncated: bool
    strategy: str
    word_count: int
    char_count: int


def _fetch_raw(url: str, *, timeout_s: float, max_bytes: int) -> tuple[str, str, int, str, bool]:
    headers = {
        "User-Agent": "deep-research-agent/0.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
        with client.stream("GET", url) as r:
            status = int(r.status_code)
            final_url = str(r.url)
            ctype = (r.headers.get("content-type") or "").lower()
            buf = bytearray()
            truncated = False
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                remaining = max_bytes - len(buf)
                if remaining <= 0:
                    truncated = True
                    break
                if len(chunk) > remaining:
                    buf.extend(chunk[:remaining])
                    truncated = True
                    break
                buf.extend(chunk)
            raw = buf.decode("utf-8", errors="replace")
            return raw, final_url, status, ctype, truncated


def fetch_document(
    url: str,
    *,
    timeout_s: float = 25.0,
    max_chars: int = 250_000,
    min_words: int = 160,
    min_chars: int = 1200,
) -> FetchResult:
    url = _validate_url(url)

    raw, final_url, status, ctype, truncated_raw = _fetch_raw(
        url,
        timeout_s=timeout_s,
        max_bytes=2_000_000,
    )

    title = extract_title(raw) if "html" in ctype else None
    extracted = raw
    strategy = "direct"

    if "html" in ctype:
        extracted = html_to_text(raw)

    wc = _word_count(extracted)
    cc = len(extracted.strip())

    if (("html" in ctype) and (wc < min_words or cc < min_chars)) or cc == 0:
        jr = _jina_reader_url(final_url or url)
        try:
            raw2, final_url2, status2, ctype2, truncated2 = _fetch_raw(
                jr,
                timeout_s=timeout_s,
                max_bytes=2_000_000,
            )
            extracted2 = raw2.strip()
            wc2 = _word_count(extracted2)
            cc2 = len(extracted2)
            if wc2 >= min_words and cc2 >= min_chars:
                extracted = extracted2
                wc = wc2
                cc = cc2
                final_url = final_url2
                status = status2
                ctype = ctype2
                truncated_raw = truncated2
                strategy = "jina"
        except Exception:
            pass

    truncated = False
    if len(extracted) > max_chars:
        extracted = extracted[:max_chars] + "\n\n[TRUNCATED]\n"
        truncated = True

    if not ctype.startswith("text/") and "html" not in ctype:
        extracted = f"Unsupported content-type: {ctype}\nURL: {final_url}\nStatus: {status}\n"
        wc = _word_count(extracted)
        cc = len(extracted.strip())
        strategy = "direct"
        truncated = False

    return FetchResult(
        ok=True,
        url=url,
        final_url=final_url,
        status_code=status,
        content_type=ctype,
        extracted_text=extracted,
        title=title,
        truncated=bool(truncated or truncated_raw),
        strategy=strategy,
        word_count=wc,
        char_count=cc,
    )


def fetch_url(url: str, *, timeout_s: float = 25.0, max_chars: int = 250_000) -> str:
    try:
        fr = fetch_document(url, timeout_s=timeout_s, max_chars=max_chars)
        return fr.extracted_text
    except Exception as e:
        return f"Fetch failed: {type(e).__name__}: {e}"