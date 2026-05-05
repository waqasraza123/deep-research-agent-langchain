from __future__ import annotations

import re
from urllib.parse import urlsplit

from deep_research_agent.tools import FetchResult

from .contracts import QualityScore

DATE_RE = re.compile(
    r"\b(?:20\d{2}|19\d{2})[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+(?:20\d{2}|19\d{2})\b"
    r"|\b(?:20\d{2}|19\d{2})\b",
    flags=re.IGNORECASE,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _authority_hint(url: str, result: FetchResult) -> float:
    parts = urlsplit(result.canonical_url or result.final_url or url)
    host = (parts.hostname or "").lower()
    path = (parts.path or "").lower()
    score = 0.35
    if host.endswith(".gov") or host.endswith(".edu"):
        score += 0.35
    docs_paths = ("/docs", "/documentation", "/reference", "/guide")
    if "docs" in host or any(p in path for p in docs_paths):
        score += 0.25
    if any(p in path for p in ("/research", "/paper", "/citation", "/source")):
        score += 0.15
    if result.kind in {"pdf", "docx", "md", "csv"}:
        score += 0.1
    return _clamp(score)


def score_source(result: FetchResult) -> QualityScore:
    warnings: list[str] = []
    if not result.ok:
        warnings.append("fetch_not_ok")

    extraction_quality = 0.0
    if result.ok:
        extraction_quality = 0.55
        if result.word_count >= 160 and result.char_count >= 1200:
            extraction_quality += 0.3
        if result.strategy == "jina":
            extraction_quality -= 0.05
        if result.truncated:
            extraction_quality -= 0.1
            warnings.append("content_truncated")
    if result.word_count < 80:
        warnings.append("very_short_content")
    elif result.word_count < 160:
        warnings.append("short_content")

    content_length_score = _clamp(result.word_count / 1200)

    text = result.extracted_text or ""
    citation_hits = len(
        re.findall(r"\b(?:doi|references|citation|cited|source|http[s]?://)\b", text, re.I)
    )
    citation_usefulness_score = _clamp(citation_hits / 8)

    matches = tuple(dict.fromkeys(m.group(0) for m in DATE_RE.finditer(text[:40_000])))
    freshness_signal = _clamp(len(matches) / 5)
    if not matches:
        warnings.append("no_visible_date")

    authority_hint = _authority_hint(result.url, result)
    final_quality_score = _clamp(
        extraction_quality * 0.35
        + content_length_score * 0.25
        + citation_usefulness_score * 0.15
        + freshness_signal * 0.1
        + authority_hint * 0.15
    )

    return QualityScore(
        extraction_quality=round(_clamp(extraction_quality), 3),
        content_length_score=round(content_length_score, 3),
        citation_usefulness_score=round(citation_usefulness_score, 3),
        freshness_signal=round(freshness_signal, 3),
        authority_hint=round(authority_hint, 3),
        final_quality_score=round(final_quality_score, 3),
        warnings=tuple(warnings),
        freshness_matches=matches[:10],
    )
