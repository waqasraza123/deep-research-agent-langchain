from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def host_domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def url_path(url: str) -> str:
    return (urlsplit(url).path or "").lower()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9.+_-]*", (value or "").lower())


def count_pattern(text: str, patterns: tuple[str, ...]) -> int:
    haystack = (text or "").lower()
    return sum(len(re.findall(pattern, haystack, flags=re.IGNORECASE)) for pattern in patterns)


def has_query_tracking(url: str) -> bool:
    tracking_prefixes = ("utm_", "fbclid", "gclid", "mc_cid", "ref", "affiliate", "aff")
    return any(k.lower().startswith(tracking_prefixes) for k, _v in parse_qsl(urlsplit(url).query))


def is_stable_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return False
    path = parts.path.lower()
    unstable_parts = (
        "/search",
        "/tag/",
        "/category/",
        "/login",
        "/signup",
        "/cart",
        "/checkout",
        "/advert",
    )
    if any(part in path for part in unstable_parts):
        return False
    if has_query_tracking(url):
        return False
    query_keys = {k.lower() for k, _v in parse_qsl(parts.query)}
    if query_keys - {"id", "p", "page", "version", "v"}:
        return False
    return True


def first_nonempty(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def text_head(text: str, limit: int = 60_000) -> str:
    return (text or "")[:limit]
