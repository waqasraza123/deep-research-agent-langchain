from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref"}


class SourceIdentity(BaseModel):
    source_id: str
    url: str
    canonical_url: str | None = None
    normalized_url: str
    domain: str | None = None
    title: str | None = None
    content_hash: str | None = None
    source_kind: str | None = None
    parent_url: str | None = None
    fetched_at: str | None = None


def stable_source_id(
    *,
    url: str,
    canonical_url: str | None = None,
    normalized_url: str | None = None,
    content_hash: str | None = None,
) -> str:
    key = content_hash or canonical_url or normalized_url or url
    digest = hashlib.sha1((key or "unknown-source").encode("utf-8")).hexdigest()[:10]
    return f"S-{digest}"


def source_domain(url: str | None) -> str | None:
    if not url:
        return None
    host = urlsplit(url).hostname
    return host.lower() if host else None


def normalize_source_url(url: str) -> str:
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
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query_items)), ""))


def source_identity_from_dict(source: dict[str, Any]) -> SourceIdentity:
    nested = source.get("source_identity")
    if isinstance(nested, dict):
        try:
            return SourceIdentity(**nested)
        except Exception:
            pass

    url = str(source.get("final_url") or source.get("url") or "")
    normalized = str(source.get("normalized_url") or normalize_source_url(url) or url)
    canonical = source.get("canonical_url")
    canonical = normalize_source_url(str(canonical)) if canonical else None
    content_digest = source.get("content_hash") or source.get("extracted_text_hash")
    content_digest = str(content_digest) if content_digest else None
    source_id = str(
        source.get("source_id")
        or source.get("id")
        or stable_source_id(
            url=url,
            canonical_url=canonical,
            normalized_url=normalized,
            content_hash=content_digest,
        )
    )
    return SourceIdentity(
        source_id=source_id,
        url=url,
        canonical_url=canonical,
        normalized_url=normalized,
        domain=source.get("source_domain") or source_domain(url),
        title=source.get("title") or source.get("source_title"),
        content_hash=content_digest,
        source_kind=source.get("source_kind"),
        parent_url=source.get("parent_url"),
        fetched_at=source.get("fetched_at"),
    )


def source_identity_to_dict(identity: SourceIdentity) -> dict[str, Any]:
    if hasattr(identity, "model_dump"):
        return identity.model_dump(mode="json")
    return identity.dict()


def enrich_source_dict(source: dict[str, Any]) -> dict[str, Any]:
    identity = source_identity_from_dict(source)
    out = dict(source)
    out["source_id"] = identity.source_id
    out["normalized_url"] = identity.normalized_url
    out["canonical_url"] = identity.canonical_url
    out["source_domain"] = identity.domain
    out["content_hash"] = identity.content_hash
    out["source_identity"] = source_identity_to_dict(identity)
    return out
