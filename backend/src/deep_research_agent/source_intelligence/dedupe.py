from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref"}


def normalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port

    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if path != "/":
        path = re.sub(r"/{2,}", "/", path).rstrip("/")

    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l in TRACKING_QUERY_KEYS or key_l.startswith(TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items))

    return urlunsplit((scheme, netloc, path, query, ""))


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def title_domain_key(title: str | None, normalized_url: str) -> str | None:
    if not title:
        return None
    host = urlsplit(normalized_url).hostname or ""
    title_norm = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    if not title_norm:
        return None
    return f"{host}|{title_norm}"


@dataclass(frozen=True)
class DedupeDecision:
    is_duplicate: bool
    duplicate_of: str | None = None
    reason: str | None = None


class DedupeIndex:
    def __init__(self) -> None:
        self._exact_urls: dict[str, str] = {}
        self._normalized_urls: dict[str, str] = {}
        self._canonical_urls: dict[str, str] = {}
        self._content_hashes: dict[str, str] = {}
        self._title_domains: dict[str, str] = {}

    @property
    def registered_count(self) -> int:
        return len(set(self._normalized_urls.values()))

    def precheck_url(self, url: str, normalized_url: str) -> DedupeDecision:
        if url in self._exact_urls:
            return DedupeDecision(True, self._exact_urls[url], "duplicate_exact_url")
        if normalized_url in self._normalized_urls:
            return DedupeDecision(
                True,
                self._normalized_urls[normalized_url],
                "duplicate_normalized_url",
            )
        return DedupeDecision(False)

    def check_fetched(
        self,
        *,
        url: str,
        normalized_url: str,
        canonical_url: str | None,
        title: str | None,
        text: str,
    ) -> DedupeDecision:
        pre = self.precheck_url(url, normalized_url)
        if pre.is_duplicate:
            return pre

        if canonical_url:
            canonical = normalize_url(canonical_url)
            if canonical in self._canonical_urls:
                return DedupeDecision(
                    True,
                    self._canonical_urls[canonical],
                    "duplicate_canonical_url",
                )

        digest = content_hash(text)
        if digest in self._content_hashes:
            return DedupeDecision(True, self._content_hashes[digest], "duplicate_content_hash")

        title_key = title_domain_key(title, normalized_url)
        if title_key and title_key in self._title_domains:
            return DedupeDecision(True, self._title_domains[title_key], "duplicate_title_domain")

        return DedupeDecision(False)

    def register(
        self,
        source_id: str,
        *,
        url: str,
        normalized_url: str,
        canonical_url: str | None,
        title: str | None,
        text: str,
    ) -> None:
        self._exact_urls.setdefault(url, source_id)
        self._normalized_urls.setdefault(normalized_url, source_id)
        if canonical_url:
            self._canonical_urls.setdefault(normalize_url(canonical_url), source_id)
        self._content_hashes.setdefault(content_hash(text), source_id)
        key = title_domain_key(title, normalized_url)
        if key:
            self._title_domains.setdefault(key, source_id)
