from __future__ import annotations

import ipaddress
import io
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


def _detect_kind(url: str, content_type: str) -> str:
    p = urlparse(url)
    path = (p.path or "").lower()
    ct = (content_type or "").lower()

    if path.endswith(".pdf") or "application/pdf" in ct:
        return "pdf"
    if path.endswith(".docx") or "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in ct:
        return "docx"
    if path.endswith(".txt") or ct.startswith("text/plain"):
        return "txt"
    if path.endswith(".md") or ct in {"text/markdown", "text/x-markdown"}:
        return "md"
    if path.endswith(".csv") or ct.startswith("text/csv"):
        return "csv"
    if "html" in ct:
        return "html"
    if ct.startswith("text/"):
        return "txt"
    return "unknown"


def _extract_pdf_text(data: bytes) -> tuple[bool, str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return False, "PDF extraction requires pypdf. Install it to enable PDF support."

    try:
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            t = t.strip()
            if t:
                parts.append(t)
        text = "\n\n".join(parts).strip()
        if text:
            return True, text
        return False, "PDF extracted with pypdf but no text was found."
    except Exception as e:
        return False, f"PDF extraction failed: {type(e).__name__}: {e}"


def _extract_docx_text(data: bytes) -> tuple[bool, str]:
    try:
        from docx import Document  # type: ignore
    except Exception:
        return False, "DOCX extraction requires python-docx. Install it to enable DOCX support."

    try:
        doc = Document(io.BytesIO(data))
        parts: list[str] = []
        for p in doc.paragraphs:
            s = (p.text or "").strip()
            if s:
                parts.append(s)
        for table in getattr(doc, "tables", []):
            for row in table.rows:
                cells = [(c.text or "").strip() for c in row.cells]
                if any(cells):
                    parts.append("\t".join(cells))
        text = "\n".join(parts).strip()
        if text:
            return True, text
        return False, "DOCX extracted with python-docx but no text was found."
    except Exception as e:
        return False, f"DOCX extraction failed: {type(e).__name__}: {e}"


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
    kind: str


def _fetch_bytes(url: str, *, timeout_s: float, max_bytes: int) -> tuple[bytes, str, int, str, bool]:
    headers = {
        "User-Agent": "deep-research-agent/0.1",
        "Accept": "*/*",
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
            return bytes(buf), final_url, status, ctype, truncated


def fetch_document(
    url: str,
    *,
    timeout_s: float = 25.0,
    max_chars: int = 250_000,
    min_words: int = 160,
    min_chars: int = 1200,
) -> FetchResult:
    url = _validate_url(url)

    max_bytes = min(12_000_000, max(128_000, max_chars * 6))
    data, final_url, status, ctype, truncated_raw = _fetch_bytes(url, timeout_s=timeout_s, max_bytes=max_bytes)

    _validate_url(final_url)

    kind = _detect_kind(final_url, ctype)

    if status < 200 or status >= 300:
        text = f"Fetch failed with status {status}\nURL: {final_url}\n"
        text = _normalize_text(text)
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
            word_count=_word_count(text),
            char_count=len(text),
            kind=kind,
        )

    title: Optional[str] = None
    extracted = ""
    ok = True
    strategy = "direct"

    if kind == "html":
        raw = data.decode("utf-8", errors="replace")
        title = extract_title(raw)
        extracted = html_to_text(raw)
        extracted = _normalize_text(extracted)
        wc = _word_count(extracted)
        cc = len(extracted)
        if (wc < min_words or cc < min_chars) or cc == 0:
            jr = _jina_reader_url(final_url)
            try:
                data2, final_url2, status2, ctype2, truncated2 = _fetch_bytes(jr, timeout_s=timeout_s, max_bytes=max_bytes)
                text2 = _normalize_text(data2.decode("utf-8", errors="replace"))
                wc2 = _word_count(text2)
                cc2 = len(text2)
                if status2 >= 200 and status2 < 300 and wc2 >= min_words and cc2 >= min_chars:
                    extracted = text2
                    wc = wc2
                    cc = cc2
                    final_url = final_url2
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
            word_count=_word_count(extracted),
            char_count=len(extracted),
            kind=kind,
        )

    if kind in {"txt", "md", "csv"}:
        extracted = data.decode("utf-8", errors="replace")
        extracted = _normalize_text(extracted)
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
            title=None,
            truncated=bool(truncated or truncated_raw),
            strategy="direct",
            word_count=_word_count(extracted),
            char_count=len(extracted),
            kind=kind,
        )

    if kind == "pdf":
        ok2, text = _extract_pdf_text(data)
        ok = bool(ok2)
        extracted = _normalize_text(text)
        truncated = False
        if len(extracted) > max_chars:
            extracted = extracted[:max_chars] + "\n\n[TRUNCATED]\n"
            truncated = True
        return FetchResult(
            ok=ok,
            url=url,
            final_url=final_url,
            status_code=status,
            content_type=ctype,
            extracted_text=extracted,
            title=None,
            truncated=bool(truncated or truncated_raw),
            strategy="direct",
            word_count=_word_count(extracted),
            char_count=len(extracted),
            kind=kind,
        )

    if kind == "docx":
        ok2, text = _extract_docx_text(data)
        ok = bool(ok2)
        extracted = _normalize_text(text)
        truncated = False
        if len(extracted) > max_chars:
            extracted = extracted[:max_chars] + "\n\n[TRUNCATED]\n"
            truncated = True
        return FetchResult(
            ok=ok,
            url=url,
            final_url=final_url,
            status_code=status,
            content_type=ctype,
            extracted_text=extracted,
            title=None,
            truncated=bool(truncated or truncated_raw),
            strategy="direct",
            word_count=_word_count(extracted),
            char_count=len(extracted),
            kind=kind,
        )

    extracted = _normalize_text(f"Unsupported content-type: {ctype}\nURL: {final_url}\n")
    return FetchResult(
        ok=False,
        url=url,
        final_url=final_url,
        status_code=status,
        content_type=ctype,
        extracted_text=extracted,
        title=None,
        truncated=False,
        strategy="direct",
        word_count=_word_count(extracted),
        char_count=len(extracted),
        kind=kind,
    )


def fetch_url(url: str, *, timeout_s: float = 25.0, max_chars: int = 250_000) -> str:
    try:
        fr = fetch_document(url, timeout_s=timeout_s, max_chars=max_chars)
        return fr.extracted_text
    except Exception as e:
        return f"Fetch failed: {type(e).__name__}: {e}"