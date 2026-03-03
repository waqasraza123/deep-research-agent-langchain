from __future__ import annotations

import ipaddress
import re
import socket
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
        t = (tag or "").lower()
        if t == "script":
            self._in_script = True
            return
        if t == "style":
            self._in_style = True
            return
        if t in {"p", "br", "div", "section", "article", "main", "li", "ul", "ol"} or t.startswith("h"):
            self.text_parts.append("\n")
        if t == "a":
            for k, v in attrs:
                if (k or "").lower() == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag):
        t = (tag or "").lower()
        if t == "script":
            self._in_script = False
        elif t == "style":
            self._in_style = False
        elif t in {"p", "li"} or t.startswith("h"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self._in_script or self._in_style:
            return
        s = (data or "").strip()
        if s:
            self.text_parts.append(s)


def _normalize_text(text: str) -> str:
    t = re.sub(r"[ \t]{2,}", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_noncontent_blocks(html: str) -> str:
    t = html
    for tag in ("nav", "header", "footer", "aside"):
        t = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            "",
            t,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return t


def _select_content_html(html: str) -> str:
    h = _strip_noncontent_blocks(html)
    for tag in ("main", "article", "body"):
        m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", h, flags=re.IGNORECASE | re.DOTALL)
        if m and m.group(1):
            return m.group(1)
    return h


def html_to_text(html: str) -> str:
    content_html = _select_content_html(html)
    parser = _TextAndLinksParser()
    parser.feed(content_html)
    return _normalize_text("\n".join(parser.text_parts))


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


def _is_ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_is_blocked(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return True
    if h in {"localhost"}:
        return True
    if h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return _is_ip_blocked(ip)
    except Exception:
        pass
    try:
        infos = socket.getaddrinfo(h, None)
        ips = {info[4][0] for info in infos if info and info[4] and info[4][0]}
        for ip_str in ips:
            try:
                ip = ipaddress.ip_address(ip_str)
                if _is_ip_blocked(ip):
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _validate_url(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("only http(s) urls allowed")
    if p.username or p.password:
        raise ValueError("userinfo not allowed")
    if not p.hostname:
        raise ValueError("missing host")
    if _host_is_blocked(p.hostname):
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
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    transport = httpx.HTTPTransport(retries=0)
    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers, limits=limits, transport=transport) as client:
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

    max_bytes = min(5_000_000, max(64_000, max_chars * 4))
    raw, final_url, status, ctype, truncated_raw = _fetch_raw(url, timeout_s=timeout_s, max_bytes=max_bytes)

    _validate_url(final_url)

    if status < 200 or status >= 300:
        text = f"Fetch failed with status {status}\nURL: {final_url}\n"
        wc = _word_count(text)
        cc = len(text.strip())
        return FetchResult(
            ok=False,
            url=url,
            final_url=final_url,
            status_code=status,
            content_type=ctype,
            extracted_text=text,
            title=None,
            truncated=False,
            strategy="direct",
            word_count=wc,
            char_count=cc,
        )

    title = extract_title(raw) if "html" in ctype else None
    extracted = raw
    strategy = "direct"

    if "html" in ctype:
        extracted = html_to_text(raw)

    extracted = _normalize_text(extracted)
    wc = _word_count(extracted)
    cc = len(extracted)

    if (("html" in ctype) and (wc < min_words or cc < min_chars)) or cc == 0:
        jr = _jina_reader_url(final_url or url)
        try:
            raw2, final_url2, status2, ctype2, truncated2 = _fetch_raw(jr, timeout_s=timeout_s, max_bytes=max_bytes)
            text2 = _normalize_text(raw2)
            wc2 = _word_count(text2)
            cc2 = len(text2)
            if status2 >= 200 and status2 < 300 and wc2 >= min_words and cc2 >= min_chars:
                extracted = text2
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