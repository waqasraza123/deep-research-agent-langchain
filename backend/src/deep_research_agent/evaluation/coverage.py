from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlparse

from deep_research_agent.evidence.contracts import EvidenceLedger

from .contracts import CoverageGap, Severity

_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.-]+|[A-Z]{2,}))*\b"
)
_STOP_ENTITIES = {
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "How",
    "The",
    "A",
    "An",
    "I",
    "We",
    "You",
}
_PRIMARY_DOMAINS = (
    ".gov",
    ".edu",
    "sec.gov",
    "who.int",
    "europa.eu",
    "fda.gov",
    "nist.gov",
    "github.com",
    "docs.",
    "developer.",
)
_OPPOSING_TERMS = (
    "counterargument",
    "opposing view",
    "on the other hand",
    "critics",
    "criticism",
    "limitation",
    "tradeoff",
    "trade-off",
    "however",
    "although",
)
_FRESHNESS_TERMS = (
    "latest",
    "current",
    "currently",
    "recent",
    "today",
    "now",
    "this year",
    "2025",
    "2026",
)


def detect_coverage_gaps(
    *,
    question: str,
    report_text: str,
    notes_text: str,
    sources: list[dict[str, Any]],
    evidence_ledger: EvidenceLedger | None = None,
    subquestions: list[str] | None = None,
) -> list[CoverageGap]:
    gaps: list[CoverageGap] = []
    combined = f"{report_text}\n{notes_text}"
    report_lower = report_text.lower()

    for subquestion in _candidate_subquestions(question, subquestions or []):
        if not _meaningfully_answered(subquestion, report_text):
            gaps.append(
                _gap(
                    "unanswered_subquestion",
                    "medium",
                    f"Subquestion appears unanswered: {subquestion}",
                    [subquestion],
                    "Add a direct answer or explicit limitation for this subquestion.",
                    ["report.md", "notes.md"],
                )
            )

    for entity in extract_entities(question):
        if entity.lower() not in combined.lower():
            gaps.append(
                _gap(
                    "missing_question_entity",
                    "high",
                    f"Question entity is missing from the report: {entity}",
                    [entity],
                    "Address this entity directly or state why it is out of scope.",
                    ["report.md"],
                )
            )

    cited_ids = set(re.findall(r"\[(S\d+)\]", report_text, flags=re.IGNORECASE))
    cited_ids |= set(re.findall(r"\b(S\d+)\b", report_text, flags=re.IGNORECASE))
    for idx, source in enumerate(sources, start=1):
        source_id = str(source.get("source_id") or source.get("id") or f"S{idx}")
        if source.get("ok") is False or source.get("skipped") is True:
            continue
        url = str(source.get("final_url") or source.get("url") or "")
        title = str(source.get("title") or "")
        domain = _domain(url)
        used_by_text = any(
            token and token.lower() in report_lower for token in (url, title, domain)
        )
        if source_id not in cited_ids and not used_by_text:
            gaps.append(
                _gap(
                    "unused_fetched_url",
                    "medium",
                    f"Fetched source appears unused: {source_id} {url}".strip(),
                    [url or source_id],
                    "Either cite/use the source or remove it from the research basis.",
                    ["sources.json", "report.md"],
                )
            )
        elif used_by_text and source_id not in cited_ids:
            gaps.append(
                _gap(
                    "source_used_not_cited",
                    "medium",
                    f"Source appears used without an explicit citation marker: {source_id}",
                    [url or title or source_id],
                    "Add a citation marker such as [S1] near the supported claim.",
                    ["report.md"],
                )
            )

    if evidence_ledger is not None:
        for unsupported in evidence_ledger.unsupported_claims:
            gaps.append(
                _gap(
                    "unsupported_claim",
                    "high",
                    "Generated claim lacks source support.",
                    [unsupported.text],
                    unsupported.reason
                    or "Back this claim with source evidence or remove/qualify it.",
                    ["report.md", "evidence_ledger.json"],
                )
            )

    if _needs_primary_source(question, report_text) and not _has_primary_source(sources):
        gaps.append(
            _gap(
                "missing_primary_source",
                "high",
                (
                    "The question appears to require primary or official sources, "
                    "but none were detected."
                ),
                [],
                "Add official documentation, regulatory filings, papers, or other primary sources.",
                ["sources.json", "report.md"],
            )
        )

    if _needs_opposing_view(question) and not any(term in report_lower for term in _OPPOSING_TERMS):
        gaps.append(
            _gap(
                "missing_opposing_view",
                "medium",
                "Comparative or controversial question lacks opposing views or limitations.",
                [],
                "Add counterarguments, tradeoffs, and limitations from credible sources.",
                ["report.md"],
            )
        )

    if is_time_sensitive(question) and not _has_freshness_verification(report_text, sources):
        gaps.append(
            _gap(
                "missing_freshness_verification",
                "high",
                "Time-sensitive question lacks clear freshness verification.",
                [],
                "State the research date and cite current dated sources.",
                ["report.md", "sources.json"],
            )
        )

    return _dedupe_gaps(gaps)


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        entity = match.group(0).strip(" ,.;:()[]")
        if len(entity) < 2 or entity in _STOP_ENTITIES:
            continue
        if entity.lower() in {"compare", "versus"}:
            continue
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            entities.append(entity)
    return entities


def is_time_sensitive(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _FRESHNESS_TERMS)


def _candidate_subquestions(question: str, provided: list[str]) -> list[str]:
    out: list[str] = [item.strip() for item in provided if item and item.strip()]
    pieces = re.split(r"\?\s+|\s+\band\b\s+|\s*;\s*", question)
    if len(pieces) > 1:
        out.extend(piece.strip(" ?") for piece in pieces if len(piece.strip()) >= 18)
    if re.search(r"\b(compare|versus|vs\.?|tradeoffs?)\b", question, flags=re.IGNORECASE):
        entities = extract_entities(question)
        out.extend(f"Evaluate {entity}" for entity in entities[:4])
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:12]


def _meaningfully_answered(subquestion: str, report_text: str) -> bool:
    terms = _content_terms(subquestion)
    if not terms:
        return True
    report_terms = set(_content_terms(report_text))
    overlap = len(set(terms) & report_terms) / max(len(set(terms)), 1)
    return overlap >= 0.45


def _content_terms(text: str) -> list[str]:
    stop = {
        "about",
        "after",
        "against",
        "also",
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "should",
        "the",
        "their",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return [
        t.lower()
        for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b", text)
        if t.lower() not in stop
    ]


def _needs_primary_source(question: str, report_text: str) -> bool:
    combined = f"{question}\n{report_text}".lower()
    return any(
        term in combined
        for term in (
            "official",
            "regulation",
            "regulatory",
            "filing",
            "law",
            "legal",
            "medical",
            "clinical",
            "api",
            "documentation",
            "benchmark",
            "release",
        )
    )


def _has_primary_source(sources: list[dict[str, Any]]) -> bool:
    for source in sources:
        url = str(source.get("final_url") or source.get("url") or "").lower()
        title = str(source.get("title") or "").lower()
        if any(domain in url for domain in _PRIMARY_DOMAINS):
            return True
        if any(term in title for term in ("official", "documentation", "paper", "filing")):
            return True
    return False


def _needs_opposing_view(question: str) -> bool:
    q = question.lower()
    return any(
        term in q
        for term in (
            "compare",
            "versus",
            " vs ",
            "controvers",
            "debate",
            "best",
            "should",
            "pros and cons",
            "tradeoff",
            "recommend",
        )
    )


def _has_freshness_verification(report_text: str, sources: list[dict[str, Any]]) -> bool:
    if re.search(r"\bas of\b|\bresearch date\b|\bfetched\b", report_text, flags=re.IGNORECASE):
        return True
    return any(source.get("fetched_at") for source in sources)


def _domain(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).hostname or ""
    return host.lower()


def _gap(
    kind: str,
    severity: Severity,
    description: str,
    evidence: list[str],
    suggested_fix: str,
    affected_artifacts: list[str],
) -> CoverageGap:
    digest = hashlib.sha1(f"{kind}:{description}:{evidence}".encode("utf-8")).hexdigest()[:10]
    return CoverageGap(
        gap_id=f"G-{digest}",
        kind=kind,  # type: ignore[arg-type]
        severity=severity,
        description=description,
        evidence=evidence,
        suggested_fix=suggested_fix,
        affected_artifacts=affected_artifacts,
    )


def _dedupe_gaps(gaps: list[CoverageGap]) -> list[CoverageGap]:
    seen: set[tuple[str, str]] = set()
    out: list[CoverageGap] = []
    for gap in gaps:
        key = (gap.kind, gap.description)
        if key in seen:
            continue
        seen.add(key)
        out.append(gap)
    return out
