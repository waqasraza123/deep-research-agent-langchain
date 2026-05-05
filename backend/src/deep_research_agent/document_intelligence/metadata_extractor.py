from __future__ import annotations

import re
from typing import Any

from deep_research_agent.source_identity import source_domain, source_identity_from_dict

from .contracts import DocumentMetadata


def _keyword_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()][:40]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;]", value) if part.strip()][:40]
    return []


def metadata_from_source(
    source: dict[str, Any], *, raw_text: str, normalized_text: str
) -> DocumentMetadata:
    return DocumentMetadata(
        author=source.get("author"),
        published_at=source.get("published_at") or source.get("date_published"),
        modified_at=source.get("modified_at") or source.get("last_modified"),
        description=source.get("description"),
        keywords=_keyword_list(source.get("keywords")),
        html_headings=_keyword_list(source.get("html_headings")),
        extraction_strategy=source.get("strategy"),
        content_type=source.get("content_type"),
        status_code=source.get("status_code"),
        canonical_url=source.get("canonical_url"),
        local_path=source.get("local_path"),
        word_count=len(re.findall(r"\b\w+\b", normalized_text or "")),
        char_count=len(normalized_text or ""),
        raw_char_count=len(raw_text or ""),
        extra={
            key: source[key]
            for key in ("source_kind", "crawl_depth", "parent_url", "priority_score", "truncated")
            if key in source
        },
    )


def source_basics(source: dict[str, Any]) -> tuple[str, str, str | None, str | None, str]:
    identity = source_identity_from_dict(source)
    source_id = identity.source_id
    url = identity.url or str(source.get("url") or "")
    title = identity.title or source.get("title")
    domain = identity.domain or source_domain(url)
    source_type = str(source.get("document_kind") or source.get("source_type") or "unknown")
    return source_id, url, title, domain, source_type
