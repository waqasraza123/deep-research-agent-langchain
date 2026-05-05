from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ._heuristics import clamp, first_nonempty, host_domain
from .authority import score_authority
from .bias import analyze_bias_risk
from .citation_readiness import score_citation_readiness
from .contracts import (
    RecommendedUsage,
    SourceAudit,
    SourceAuditBatch,
    SourceAuditSummary,
    SourceAuditWarning,
)
from .credibility import score_credibility
from .freshness import score_freshness
from .primary_source import assess_primary_source


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _safe_source_text(source: dict[str, Any], thread_dir: Path | None) -> str:
    for key in ("text", "content", "extracted_text"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if thread_dir is None:
        return ""
    local_path = source.get("local_path")
    if not isinstance(local_path, str) or not local_path:
        return ""
    try:
        name = Path(local_path).name
        if not name or name in {".", ".."}:
            return ""
        path = (thread_dir / "sources" / name).resolve()
        if not str(path).startswith(str(thread_dir.resolve())):
            return ""
        if path.exists() and not path.is_dir():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def _source_type(source: dict[str, Any], url: str) -> str:
    value = first_nonempty(
        source.get("document_kind"),
        source.get("kind"),
        source.get("source_type"),
    )
    if value:
        return value
    content_type = (source.get("content_type") or "").lower()
    path = (urlsplit(url).path or "").lower()
    if path.endswith(".pdf") or "pdf" in content_type:
        return "pdf"
    if path.endswith(".csv") or "csv" in content_type:
        return "csv"
    if path.endswith(".md"):
        return "md"
    if "html" in content_type:
        return "html"
    return "unknown"


def _warnings_for_source(
    source: dict[str, Any],
    *,
    text: str,
    word_count: int,
) -> list[SourceAuditWarning]:
    warnings: list[SourceAuditWarning] = []
    if source.get("ok") is False:
        warnings.append(
            SourceAuditWarning(
                code="fetch_not_ok",
                severity="high",
                message="Fetch did not complete successfully.",
            )
        )
    if source.get("skipped") is True:
        warnings.append(
            SourceAuditWarning(
                code="source_skipped",
                severity="high",
                message=f"Source was skipped: {source.get('skip_reason') or 'unknown reason'}.",
            )
        )
    if source.get("duplicate_of"):
        warnings.append(
            SourceAuditWarning(
                code="duplicate_source",
                severity="medium",
                message=f"Source duplicates {source.get('duplicate_of')}.",
            )
        )
    if source.get("truncated") is True:
        warnings.append(
            SourceAuditWarning(
                code="content_truncated",
                severity="medium",
                message="Fetched content was truncated.",
            )
        )
    if word_count < 80:
        warnings.append(
            SourceAuditWarning(
                code="very_short_content",
                severity="high",
                message="Extracted content is too short for reliable citation.",
            )
        )
    elif word_count < 160:
        warnings.append(
            SourceAuditWarning(
                code="short_content",
                severity="medium",
                message="Extracted content is short.",
            )
        )
    quality = source.get("quality_score")
    if isinstance(quality, dict):
        for item in quality.get("warnings") or []:
            if isinstance(item, str):
                warnings.append(
                    SourceAuditWarning(
                        code=f"quality_{item}",
                        severity="low",
                        message=f"Source quality warning: {item}.",
                    )
                )
    if not text.strip():
        warnings.append(
            SourceAuditWarning(
                code="missing_source_text",
                severity="high",
                message="No extracted source text was available to audit.",
            )
        )
    return warnings


def _recommended_usage(
    *,
    final_score: float,
    citation_ready: bool,
    credibility: float,
    freshness_score: float,
    freshness_matters: bool,
    authority: float,
    primary_likelihood: float,
    bias_risk: float,
    high_warning_count: int,
) -> RecommendedUsage:
    if high_warning_count > 0 or final_score < 0.18:
        return "exclude_from_report"
    if freshness_matters and freshness_score < 0.45:
        return "verify_with_primary_source"
    if bias_risk >= 0.62:
        return "use_with_caution"
    if citation_ready and final_score >= 0.72 and credibility >= 0.62 and authority >= 0.55:
        return "cite_directly"
    if primary_likelihood >= 0.65 and credibility >= 0.5:
        return "verify_with_primary_source" if not citation_ready else "cite_directly"
    if final_score >= 0.48:
        return "use_as_background"
    return "use_with_caution"


def audit_source(
    source: dict[str, Any],
    *,
    question: str,
    thread_dir: Path | None = None,
    now: datetime | None = None,
) -> SourceAudit:
    url = first_nonempty(source.get("final_url"), source.get("canonical_url"), source.get("url"))
    title = first_nonempty(source.get("title")) or None
    source_id = (
        first_nonempty(source.get("source_id")) or f"S-{hashlib.sha1(url.encode()).hexdigest()[:8]}"
    )
    domain = host_domain(url)
    text = _safe_source_text(source, thread_dir)
    word_count = int(source.get("word_count") or _word_count(text))
    source_type = _source_type(source, url)
    metadata = {k: v for k, v in source.items() if k not in {"text", "content", "extracted_text"}}

    warnings = _warnings_for_source(source, text=text, word_count=word_count)
    credibility = score_credibility(url=url, title=title, text=text, metadata=metadata)
    freshness = score_freshness(
        question=question,
        url=url,
        title=title,
        text=text,
        metadata=metadata,
        now=now,
    )
    authority = score_authority(url=url, title=title, text=text, source_type=source_type)
    primary = assess_primary_source(url=url, title=title, text=text, source_type=source_type)
    bias = analyze_bias_risk(url=url, title=title, text=text)
    citation = score_citation_readiness(
        url=url,
        title=title,
        word_count=word_count,
        freshness=freshness,
        authority=authority,
        primary=primary,
        warnings=warnings,
        duplicate_of=source.get("duplicate_of"),
    )
    high_warning_count = len([w for w in warnings if w.severity == "high"])

    final = clamp(
        credibility.score * 0.25
        + freshness.score * 0.16
        + authority.score * 0.21
        + primary.likelihood * 0.16
        + citation.score * 0.16
        + (1.0 - bias.score) * 0.06
    )
    if high_warning_count:
        final *= 0.45
    elif any(w.severity == "medium" for w in warnings):
        final *= 0.85

    usage = _recommended_usage(
        final_score=final,
        citation_ready=citation.citation_ready,
        credibility=credibility.score,
        freshness_score=freshness.score,
        freshness_matters=freshness.freshness_matters,
        authority=authority.score,
        primary_likelihood=primary.likelihood,
        bias_risk=bias.score,
        high_warning_count=high_warning_count,
    )
    reasons = [
        f"Credibility {credibility.score:.3f}; authority {authority.score:.3f}; "
        f"freshness {freshness.status} ({freshness.score:.3f}).",
        f"Primary-source likelihood {primary.likelihood:.3f}; bias risk {bias.risk_level}.",
        f"Citation readiness {citation.score:.3f}; recommended usage: {usage}.",
    ]
    return SourceAudit(
        source_id=source_id,
        url=url,
        domain=domain,
        title=title,
        source_type=source_type,
        credibility_score=credibility,
        freshness_score=freshness,
        authority_score=authority,
        bias_risk_score=bias,
        primary_source_likelihood=primary,
        citation_readiness_score=citation,
        final_source_score=round(clamp(final), 3),
        warnings=warnings,
        reasons=reasons,
        recommended_usage=usage,
        metadata={
            "word_count": word_count,
            "char_count": int(source.get("char_count") or len(text)),
            "source_kind": source.get("source_kind"),
            "local_path": source.get("local_path"),
            "duplicate_of": source.get("duplicate_of"),
        },
    )


def build_source_audit_instruction_block(batch: SourceAuditBatch) -> str:
    ranked = sorted(batch.audits, key=lambda item: item.final_source_score, reverse=True)
    primary = [a for a in ranked if a.recommended_usage == "cite_directly"][:5]
    cautious = [
        a
        for a in ranked
        if a.recommended_usage in {"use_with_caution", "verify_with_primary_source"}
    ][:5]
    stale = [
        a
        for a in ranked
        if a.freshness_score.status in {"possibly_stale", "stale"}
        or (a.freshness_score.freshness_matters and a.freshness_score.status == "unknown")
    ][:5]
    lines = ["Source audit guidance for report generation:"]
    if primary:
        lines.append(
            "Prioritize direct citations: "
            + "; ".join(f"{a.source_id} ({a.domain}, {a.final_source_score:.2f})" for a in primary)
            + "."
        )
    if cautious:
        lines.append(
            "Use with caution or verify: "
            + "; ".join(
                f"{a.source_id} ({a.recommended_usage}, bias {a.bias_risk_score.risk_level})"
                for a in cautious
            )
            + "."
        )
    if stale:
        lines.append(
            "Freshness risks: "
            + "; ".join(f"{a.source_id} ({a.freshness_score.status})" for a in stale)
            + "."
        )
    if batch.summary.coverage_gaps:
        lines.append("Coverage gaps: " + "; ".join(batch.summary.coverage_gaps) + ".")
    if batch.summary.authority_gaps:
        lines.append("Authority gaps: " + "; ".join(batch.summary.authority_gaps) + ".")
    if batch.summary.citation_risks:
        lines.append("Citation risks: " + "; ".join(batch.summary.citation_risks[:4]) + ".")
    lines.append(
        "Do not cite excluded sources directly; use them only to discover primary sources."
    )
    return "\n".join(lines)


def _build_summary(audits: list[SourceAudit]) -> SourceAuditSummary:
    ranked = sorted(audits, key=lambda item: item.final_source_score, reverse=True)
    usable = [a for a in audits if a.recommended_usage != "exclude_from_report"]
    primary = [
        a.source_id
        for a in ranked
        if a.primary_source_likelihood.source_role == "primary"
        and a.recommended_usage in {"cite_directly", "verify_with_primary_source"}
    ]
    needs_verification = [
        a.source_id
        for a in ranked
        if a.recommended_usage in {"verify_with_primary_source", "use_with_caution"}
    ]
    avoid = [a.source_id for a in ranked if a.recommended_usage == "exclude_from_report"]
    avg = sum(a.final_source_score for a in audits) / len(audits) if audits else 0.0

    coverage_gaps: list[str] = []
    freshness_gaps: list[str] = []
    authority_gaps: list[str] = []
    citation_risks: list[str] = []

    if not primary:
        coverage_gaps.append("No strong primary source was found.")
    if not any(a.source_type in {"pdf", "csv", "docx"} for a in audits):
        coverage_gaps.append("No document or dataset-style source was found.")
    if any(a.freshness_score.freshness_matters for a in audits) and not any(
        a.freshness_score.status in {"current", "recent"} for a in audits
    ):
        freshness_gaps.append("Freshness-sensitive question lacks a current or recent source.")
    if any(a.freshness_score.status == "unknown" for a in audits):
        freshness_gaps.append("One or more sources lack a reliable visible date.")
    if not any(a.authority_score.score >= 0.65 for a in audits):
        authority_gaps.append(
            "No high-authority institutional, official, or standards source found."
        )
    for audit in ranked:
        if not audit.citation_readiness_score.citation_ready:
            joined = ", ".join(audit.citation_readiness_score.blockers[:2]) or "not citation-ready"
            citation_risks.append(f"{audit.source_id}: {joined}")

    summary = SourceAuditSummary(
        source_count=len(audits),
        usable_source_count=len(usable),
        average_final_score=round(clamp(avg), 3),
        ranked_source_ids=[a.source_id for a in ranked],
        recommended_primary_sources=primary[:8],
        sources_needing_verification=needs_verification[:12],
        sources_to_avoid=avoid[:12],
        coverage_gaps=coverage_gaps,
        freshness_gaps=freshness_gaps,
        authority_gaps=authority_gaps,
        citation_risks=citation_risks[:12],
    )
    return summary


def audit_sources(
    sources: list[dict[str, Any]],
    *,
    question: str,
    thread_id: str | None = None,
    thread_dir: Path | None = None,
    now: datetime | None = None,
) -> SourceAuditBatch:
    audits = [
        audit_source(source, question=question, thread_dir=thread_dir, now=now)
        for source in sources
        if isinstance(source, dict)
    ]
    summary = _build_summary(audits)
    batch = SourceAuditBatch(
        thread_id=thread_id,
        question=question,
        generated_at=_now_iso_utc(),
        audits=audits,
        summary=summary,
    )
    summary.instruction_block = build_source_audit_instruction_block(batch)
    return batch


def audit_sources_from_manifest(
    manifest_path: Path,
    *,
    question: str,
    thread_id: str | None = None,
    now: datetime | None = None,
) -> SourceAuditBatch:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        data = []
    sources = data if isinstance(data, list) else []
    return audit_sources(
        sources,
        question=question,
        thread_id=thread_id,
        thread_dir=manifest_path.parent,
        now=now,
    )
