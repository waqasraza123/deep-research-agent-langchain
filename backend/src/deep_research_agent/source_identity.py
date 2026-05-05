from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field

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


class DocumentIdentity(BaseModel):
    document_id: str
    source_id: str
    url: str = ""
    title: str | None = None
    domain: str | None = None
    content_hash: str | None = None
    local_path: str | None = None


class ChunkIdentity(BaseModel):
    chunk_id: str
    document_id: str
    source_id: str
    content_hash: str
    start_offset: int = 0
    end_offset: int = 0
    ordinal: int = 0


class ArtifactIdentity(BaseModel):
    artifact_id: str
    artifact_path: str
    artifact_type: str = "artifact"
    content_hash: str | None = None
    producer_subsystem: str | None = None


class ClaimIdentity(BaseModel):
    claim_id: str
    normalized_text_hash: str
    origin: str = ""
    origin_ref: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class HypothesisIdentity(BaseModel):
    hypothesis_id: str
    normalized_text_hash: str
    hypothesis_type: str = "unknown"
    origin: str = ""


class WarningSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResearchWarning(BaseModel):
    subsystem: str
    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.MEDIUM
    affected_artifacts: list[str] = Field(default_factory=list)
    affected_sources: list[str] = Field(default_factory=list)
    recommended_action: str = ""


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


def stable_document_id(
    *,
    source_id: str,
    content_hash: str | None = None,
    url: str | None = None,
) -> str:
    key = content_hash or url or source_id or "unknown-document"
    digest = hashlib.sha1(f"{source_id}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"D-{digest}"


def stable_chunk_id(
    *,
    document_id: str,
    source_id: str,
    content_hash: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
) -> str:
    key = f"{document_id}|{source_id}|{content_hash}|{start_offset}|{end_offset}|{ordinal}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:14]
    return f"C-{digest}"


def stable_artifact_id(*, artifact_path: str, content_hash: str | None = None) -> str:
    digest = hashlib.sha1(f"{artifact_path}|{content_hash or ''}".encode("utf-8")).hexdigest()[:12]
    return f"A-{digest}"


def stable_claim_id(*, normalized_text: str, origin: str = "", ordinal: int = 0) -> str:
    digest = hashlib.sha1(f"{origin}|{ordinal}|{normalized_text}".encode("utf-8")).hexdigest()[:12]
    return f"CL-{digest}"


def stable_hypothesis_id(*, normalized_text: str, origin: str = "", ordinal: int = 0) -> str:
    digest = hashlib.sha1(f"{origin}|{ordinal}|{normalized_text}".encode("utf-8")).hexdigest()[:12]
    return f"H-{digest}"


def text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


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


def document_identity_from_source(
    source: dict[str, Any],
    *,
    content_hash: str | None = None,
) -> DocumentIdentity:
    source_identity = source_identity_from_dict(source)
    digest = content_hash or source_identity.content_hash
    return DocumentIdentity(
        document_id=stable_document_id(
            source_id=source_identity.source_id,
            content_hash=digest,
            url=source_identity.url,
        ),
        source_id=source_identity.source_id,
        url=source_identity.url,
        title=source_identity.title,
        domain=source_identity.domain,
        content_hash=digest,
        local_path=source.get("local_path"),
    )


def chunk_identity_to_dict(identity: ChunkIdentity) -> dict[str, Any]:
    if hasattr(identity, "model_dump"):
        return identity.model_dump(mode="json")
    return identity.dict()


def document_identity_to_dict(identity: DocumentIdentity) -> dict[str, Any]:
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
