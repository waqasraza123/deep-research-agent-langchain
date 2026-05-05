from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from deep_research_agent.tools import FetchResult, _validate_url

from .contracts import LinkCandidate
from .dedupe import normalize_url

UNSUPPORTED_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "sms:")
UNSUPPORTED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".dmg",
    ".exe",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mov",
    ".mp3",
    ".mp4",
    ".png",
    ".rar",
    ".svg",
    ".tar",
    ".webm",
    ".webp",
    ".zip",
}


@dataclass(frozen=True)
class LinkExtractionAttempt:
    href: str
    anchor_text: str
    candidate: LinkCandidate | None = None
    skip_reason: str | None = None


def normalize_and_validate_link(href: str, base_url: str) -> tuple[str | None, str | None]:
    raw = (href or "").strip()
    if not raw:
        return None, "empty_url"
    if raw.lower().startswith(UNSUPPORTED_SCHEMES):
        return None, "unsupported_scheme"

    absolute = urljoin(base_url, raw)
    parts = urlsplit(absolute)
    absolute = absolute.split("#", 1)[0]

    path_l = (parts.path or "").lower()
    if any(path_l.endswith(ext) for ext in UNSUPPORTED_EXTENSIONS):
        return None, "unsupported_file_type"

    try:
        _validate_url(absolute)
    except Exception as exc:
        return None, f"unsafe_url:{exc}"

    return normalize_url(absolute), None


def link_candidate(
    href: str,
    *,
    base_url: str,
    parent_url: str,
    anchor_text: str = "",
    crawl_depth: int = 1,
    source_format: str = "html",
) -> tuple[LinkCandidate | None, str | None]:
    normalized, reason = normalize_and_validate_link(href, base_url)
    if not normalized:
        return None, reason
    return (
        LinkCandidate(
            url=normalized,
            normalized_url=normalized,
            parent_url=parent_url,
            anchor_text=re.sub(r"\s+", " ", anchor_text or "").strip(),
            crawl_depth=crawl_depth,
            source_format=source_format,
        ),
        None,
    )


def extract_markdown_links(markdown: str, base_url: str, parent_url: str) -> list[LinkCandidate]:
    out: list[LinkCandidate] = []
    seen: set[str] = set()
    for match in re.finditer(r"\[([^\]]{1,200})\]\(([^)\s]+)\)", markdown or ""):
        candidate, reason = link_candidate(
            match.group(2),
            base_url=base_url,
            parent_url=parent_url,
            anchor_text=match.group(1),
            source_format="markdown",
        )
        if reason or candidate is None or candidate.normalized_url in seen:
            continue
        seen.add(candidate.normalized_url)
        out.append(candidate)
    return out


def candidates_from_fetch_result(result: FetchResult, parent_url: str) -> list[LinkCandidate]:
    return [
        attempt.candidate
        for attempt in extraction_attempts_from_fetch_result(result, parent_url)
        if attempt.candidate is not None
    ]


def extraction_attempts_from_fetch_result(
    result: FetchResult,
    parent_url: str,
) -> list[LinkExtractionAttempt]:
    base_url = result.canonical_url or result.final_url or result.url
    if result.kind == "md":
        markdown_out: list[LinkExtractionAttempt] = []
        markdown_seen: set[str] = set()
        for match in re.finditer(r"\[([^\]]{1,200})\]\(([^)\s]+)\)", result.extracted_text or ""):
            candidate, reason = link_candidate(
                match.group(2),
                base_url=base_url,
                parent_url=parent_url,
                anchor_text=match.group(1),
                source_format="markdown",
            )
            if candidate is not None and candidate.normalized_url in markdown_seen:
                reason = "duplicate_extracted_link"
                candidate = None
            if candidate is not None:
                markdown_seen.add(candidate.normalized_url)
            markdown_out.append(
                LinkExtractionAttempt(
                    href=match.group(2),
                    anchor_text=match.group(1),
                    candidate=candidate,
                    skip_reason=reason,
                )
            )
        return markdown_out
    if result.kind != "html":
        return []

    out: list[LinkExtractionAttempt] = []
    seen: set[str] = set()
    for item in result.extracted_links:
        candidate, reason = link_candidate(
            item.get("url", ""),
            base_url=base_url,
            parent_url=parent_url,
            anchor_text=item.get("anchor_text", ""),
            source_format="html",
        )
        if candidate is not None and candidate.normalized_url in seen:
            reason = "duplicate_extracted_link"
            candidate = None
        if candidate is not None:
            seen.add(candidate.normalized_url)
        out.append(
            LinkExtractionAttempt(
                href=item.get("url", ""),
                anchor_text=item.get("anchor_text", ""),
                candidate=candidate,
                skip_reason=reason,
            )
        )
    return out
