from __future__ import annotations

import hashlib
import re

from .contracts import DocumentCitation, DocumentFootnote


def _context(text: str, start: int, end: int, *, window: int = 120) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - window) : min(len(text), end + window)]).strip()


def _citation_id(source_id: str, citation_type: str, start: int, value: str) -> str:
    digest = hashlib.sha1(
        f"{source_id}|{citation_type}|{start}|{value}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{source_id}-cit-{digest}"


def extract_citations(
    text: str, *, source_id: str
) -> tuple[list[DocumentCitation], list[DocumentFootnote]]:
    citations: list[DocumentCitation] = []
    footnotes: list[DocumentFootnote] = []
    patterns: list[tuple[str, re.Pattern[str], float]] = [
        ("url", re.compile(r"https?://[^\s)\]>\"']+", re.I), 0.96),
        ("doi", re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I), 0.94),
        ("numbered_reference", re.compile(r"^\s*\[(\d{1,3})\]\s+(.{20,240})$", re.M), 0.82),
        (
            "academic_reference",
            re.compile(
                r"^\s*[A-Z][A-Za-z' -]+,\s+[A-Z](?:\.[A-Z])?\.?.{10,180}\((?:19|20)\d{2}\).*$",
                re.M,
            ),
            0.72,
        ),
        (
            "legal_citation",
            re.compile(r"\b\d+\s+U\.S\.C\.?\s+§+\s*[\w.-]+|\b\d+\s+F\.\d+d\s+\d+\b"),
            0.82,
        ),
        ("see_also", re.compile(r"\bsee also\b.{0,180}", re.I), 0.7),
    ]
    seen: set[tuple[str, int, int]] = set()
    for citation_type, pattern, confidence in patterns:
        for match in pattern.finditer(text):
            value = match.group(0).strip().rstrip(".,;")
            key = (citation_type, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                DocumentCitation(
                    citation_id=_citation_id(source_id, citation_type, match.start(), value),
                    source_id=source_id,
                    citation_type=citation_type,
                    text=value,
                    normalized_value=value.lower() if citation_type == "doi" else value,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    context=_context(text, match.start(), match.end()),
                    confidence_score=confidence,
                )
            )

    for match in re.finditer(r"(?m)^\s*(?:\[(\d{1,3})\]|\^(\w+)|(\d{1,3})\.)\s+(.{8,400})$", text):
        marker = match.group(1) or match.group(2) or match.group(3) or ""
        body = match.group(4).strip()
        if len(body.split()) < 3:
            continue
        digest = hashlib.sha1(
            f"{source_id}|fn|{match.start()}|{marker}".encode("utf-8")
        ).hexdigest()[:12]
        footnotes.append(
            DocumentFootnote(
                footnote_id=f"{source_id}-fn-{digest}",
                source_id=source_id,
                marker=marker,
                text=body,
                start_offset=match.start(),
                end_offset=match.end(),
                context=_context(text, match.start(), match.end()),
                confidence_score=0.72,
            )
        )

    citations.sort(key=lambda item: item.start_offset)
    footnotes.sort(key=lambda item: item.start_offset)
    return citations, footnotes
