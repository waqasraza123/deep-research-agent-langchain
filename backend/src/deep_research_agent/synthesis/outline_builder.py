from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from deep_research_agent.evidence.claim_extractor import (
    ClaimInput,
    extract_claims,
    normalize_claim_text,
    tokenize,
)

from .contracts import FindingCluster, ResearchFinding, SynthesisInput

_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "because",
    "between",
    "could",
    "from",
    "have",
    "into",
    "more",
    "must",
    "only",
    "over",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "with",
    "without",
}


def extract_findings(synthesis_input: SynthesisInput) -> list[ResearchFinding]:
    findings: list[ResearchFinding] = []
    seen: set[str] = set()

    ledger = synthesis_input.evidence_ledger or {}
    for raw_claim in _as_list(ledger.get("claims")):
        if not isinstance(raw_claim, dict):
            continue
        text = _clean_text(str(raw_claim.get("text") or ""))
        if not text:
            continue
        normalized = str(raw_claim.get("normalized_text") or normalize_claim_text(text))
        if normalized in seen:
            continue
        seen.add(normalized)
        contradiction_ids = [
            str(item) for item in raw_claim.get("contradiction_ids") or [] if str(item).strip()
        ]
        support_level = str(raw_claim.get("support_level") or "unknown")
        source_ids = [str(item) for item in raw_claim.get("source_ids") or [] if str(item).strip()]
        citations = [item for item in raw_claim.get("citations") or [] if isinstance(item, dict)]
        for citation in citations:
            source_id = citation.get("source_id")
            if source_id and str(source_id) not in source_ids:
                source_ids.append(str(source_id))
        finding = ResearchFinding(
            finding_id=_finding_id("evidence", normalized, len(findings) + 1),
            text=text,
            normalized_text=normalized,
            origin="evidence_ledger",
            origin_ref=str(raw_claim.get("claim_id") or ""),
            artifact_refs=["evidence_ledger.json"],
            source_ids=sorted(source_ids),
            topics=_topics_for(text, synthesis_input.question),
            subquestion_ids=_match_subquestions(text, synthesis_input.subquestions),
            entities=_entities_for(text, synthesis_input.question),
            claim_type=_coerce_claim_type(str(raw_claim.get("claim_type") or "factual")),
            confidence_label=_coerce_confidence(support_level),
            confidence_score=_optional_float(raw_claim.get("confidence_score")),
            contradiction_status="contradicted" if contradiction_ids else "none",
            contradiction_ids=contradiction_ids,
            citations=citations,
            notes=[str(item) for item in raw_claim.get("notes") or [] if str(item).strip()],
            requires_human_review=bool(raw_claim.get("needs_human_review", True)),
        )
        findings.append(finding)

    if not findings:
        findings.extend(_extract_from_text_artifact(synthesis_input, "notes", "notes.md"))
        findings.extend(_extract_from_text_artifact(synthesis_input, "report", "report.md"))
    else:
        findings.extend(
            _extract_missing_text_findings(
                synthesis_input=synthesis_input,
                origin="notes",
                artifact_ref="notes.md",
                existing=seen,
            )
        )
        findings.extend(
            _extract_missing_text_findings(
                synthesis_input=synthesis_input,
                origin="report",
                artifact_ref="report.md",
                existing=seen,
            )
        )

    current_seen = seen | {f.normalized_text for f in findings}
    findings.extend(_extract_source_summary_findings(synthesis_input, current_seen))
    findings.extend(
        _extract_strategy_findings(synthesis_input, {f.normalized_text for f in findings})
    )
    findings.extend(
        _extract_source_audit_findings(synthesis_input, {f.normalized_text for f in findings})
    )
    return findings


def cluster_findings(findings: list[ResearchFinding]) -> list[FindingCluster]:
    clusters: list[FindingCluster] = []
    clusters.extend(_clusters_by_multi_value("topic", findings, lambda item: item.topics))
    clusters.extend(
        _clusters_by_multi_value("subquestion", findings, lambda item: item.subquestion_ids)
    )
    clusters.extend(_clusters_by_multi_value("entity", findings, lambda item: item.entities))
    clusters.extend(_clusters_by_multi_value("source", findings, lambda item: item.source_ids))
    clusters.extend(_clusters_by_single("claim_type", findings, lambda item: item.claim_type))
    clusters.extend(_clusters_by_single("confidence", findings, lambda item: item.confidence_label))
    clusters.extend(
        _clusters_by_single("contradiction", findings, lambda item: item.contradiction_status)
    )
    return clusters


def _extract_from_text_artifact(
    synthesis_input: SynthesisInput,
    origin: str,
    artifact_ref: str,
) -> list[ResearchFinding]:
    existing: set[str] = set()
    return _extract_missing_text_findings(
        synthesis_input=synthesis_input,
        origin=origin,
        artifact_ref=artifact_ref,
        existing=existing,
    )


def _extract_missing_text_findings(
    *,
    synthesis_input: SynthesisInput,
    origin: str,
    artifact_ref: str,
    existing: set[str],
) -> list[ResearchFinding]:
    text = synthesis_input.notes_text if origin == "notes" else synthesis_input.report_text
    extracted = extract_claims(
        [ClaimInput(origin=origin, text=text, origin_ref=artifact_ref)]  # type: ignore[arg-type]
    )
    findings: list[ResearchFinding] = []
    for claim in extracted:
        if claim.normalized_text in existing:
            continue
        existing.add(claim.normalized_text)
        finding = ResearchFinding(
            finding_id=_finding_id(origin, claim.normalized_text, len(findings) + 1),
            text=claim.text,
            normalized_text=claim.normalized_text,
            origin=origin,  # type: ignore[arg-type]
            origin_ref=artifact_ref,
            artifact_refs=[artifact_ref],
            source_ids=claim.source_ids,
            topics=_topics_for(claim.text, synthesis_input.question),
            subquestion_ids=_match_subquestions(claim.text, synthesis_input.subquestions),
            entities=_entities_for(claim.text, synthesis_input.question),
            claim_type=_coerce_claim_type(claim.claim_type),
            confidence_label="unsupported",
            confidence_score=None,
            notes=claim.notes,
            requires_human_review=True,
        )
        findings.append(finding)
    return findings


def _extract_source_summary_findings(
    synthesis_input: SynthesisInput,
    existing: set[str],
) -> list[ResearchFinding]:
    findings: list[ResearchFinding] = []
    for idx, source in enumerate(synthesis_input.sources, start=1):
        summary = _clean_text(str(source.get("summary") or source.get("description") or ""))
        if not summary:
            continue
        source_id = str(source.get("source_id") or source.get("id") or f"S{idx}")
        normalized = normalize_claim_text(summary)
        if not normalized or normalized in existing:
            continue
        existing.add(normalized)
        findings.append(
            ResearchFinding(
                finding_id=_finding_id("source_summary", normalized, idx),
                text=summary,
                normalized_text=normalized,
                origin="source_summary",
                origin_ref=source_id,
                artifact_refs=["sources.json"],
                source_ids=[source_id],
                topics=_topics_for(summary, synthesis_input.question),
                subquestion_ids=_match_subquestions(summary, synthesis_input.subquestions),
                entities=_entities_for(summary, synthesis_input.question),
                claim_type="factual",
                confidence_label="weak" if source.get("ok", True) else "unsupported",
                confidence_score=None,
                notes=["Derived from source metadata summary."],
                requires_human_review=True,
            )
        )
    return findings


def _extract_strategy_findings(
    synthesis_input: SynthesisInput,
    existing: set[str],
) -> list[ResearchFinding]:
    findings: list[ResearchFinding] = []
    for idx, subquestion in enumerate(synthesis_input.subquestions, start=1):
        text = _clean_text(str(subquestion.get("question") or ""))
        if not text:
            continue
        normalized = normalize_claim_text(text)
        if normalized in existing:
            continue
        existing.add(normalized)
        sq_id = str(subquestion.get("id") or f"SQ{idx}")
        findings.append(
            ResearchFinding(
                finding_id=_finding_id("strategy", normalized, idx),
                text=text,
                normalized_text=normalized,
                origin="strategy",
                origin_ref=sq_id,
                artifact_refs=["subquestions.json"],
                topics=_topics_for(text, synthesis_input.question),
                subquestion_ids=[sq_id],
                entities=_entities_for(text, synthesis_input.question),
                claim_type="question",
                confidence_label="unknown",
                notes=["Research strategy subquestion, not a verified finding."],
                requires_human_review=True,
            )
        )
    return findings


def _extract_source_audit_findings(
    synthesis_input: SynthesisInput,
    existing: set[str],
) -> list[ResearchFinding]:
    text = synthesis_input.source_audit_text
    if not text and synthesis_input.source_audit:
        text = " ".join(str(v) for v in synthesis_input.source_audit.values() if isinstance(v, str))
    if not text:
        return []
    claims = extract_claims([ClaimInput(origin="report", text=text, origin_ref="source_audit")])
    findings: list[ResearchFinding] = []
    for idx, claim in enumerate(claims, start=1):
        if claim.normalized_text in existing:
            continue
        existing.add(claim.normalized_text)
        findings.append(
            ResearchFinding(
                finding_id=_finding_id("source_audit", claim.normalized_text, idx),
                text=claim.text,
                normalized_text=claim.normalized_text,
                origin="source_audit",
                origin_ref="source_audit",
                artifact_refs=[
                    "source_audit.json" if synthesis_input.source_audit else "source_audit.md"
                ],
                topics=_topics_for(claim.text, synthesis_input.question),
                subquestion_ids=_match_subquestions(claim.text, synthesis_input.subquestions),
                entities=_entities_for(claim.text, synthesis_input.question),
                claim_type=_coerce_claim_type(claim.claim_type),
                confidence_label="weak",
                notes=["Derived from source audit artifact."],
                requires_human_review=True,
            )
        )
    return findings


def _clusters_by_multi_value(kind: str, findings, getter) -> list[FindingCluster]:
    groups: dict[str, list[str]] = defaultdict(list)
    for finding in findings:
        for value in getter(finding):
            if value:
                groups[str(value)].append(finding.finding_id)
    return [_cluster(kind, label, ids) for label, ids in sorted(groups.items())]


def _clusters_by_single(kind: str, findings, getter) -> list[FindingCluster]:
    groups: dict[str, list[str]] = defaultdict(list)
    for finding in findings:
        value = str(getter(finding) or "unknown")
        groups[value].append(finding.finding_id)
    return [_cluster(kind, label, ids) for label, ids in sorted(groups.items())]


def _cluster(kind: str, label: str, finding_ids: list[str]) -> FindingCluster:
    return FindingCluster(
        cluster_id=_short_id(f"{kind}:{label}"),
        cluster_kind=kind,  # type: ignore[arg-type]
        label=label,
        finding_ids=sorted(finding_ids),
        representative_finding_ids=sorted(finding_ids)[:3],
        summary=f"{len(finding_ids)} finding(s) grouped by {kind}: {label}.",
        metadata={"count": len(finding_ids)},
    )


def _topics_for(text: str, question: str) -> list[str]:
    tokens = [t for t in tokenize(question + " " + text) if t not in _STOPWORDS and len(t) >= 4]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda item: (-counts[item], item))
    return ranked[:5] or ["general"]


def _entities_for(text: str, question: str) -> list[str]:
    candidates = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9+.#/-]{1,}|[a-zA-Z0-9_.+-]+(?:AI|API|DB|SQL|Graph|CPP|\.cpp))\b",
        question + " " + text,
    )
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = candidate.strip(".,:;()[]")
        if len(clean) < 2 or clean.lower() in _STOPWORDS:
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(clean)
    return out[:8]


def _match_subquestions(text: str, subquestions: list[dict[str, Any]]) -> list[str]:
    text_tokens = set(tokenize(text))
    matches: list[str] = []
    for idx, sq in enumerate(subquestions, start=1):
        sq_text = str(sq.get("question") or "")
        sq_tokens = {token for token in tokenize(sq_text) if token not in _STOPWORDS}
        if sq_tokens and len(text_tokens & sq_tokens) >= min(2, len(sq_tokens)):
            matches.append(str(sq.get("id") or f"SQ{idx}"))
    return matches


def _coerce_claim_type(value: str) -> str:
    allowed = {
        "factual",
        "comparative",
        "numeric",
        "date_sensitive",
        "causal",
        "recommendation",
        "unsupported_broad",
        "assumption",
        "risk",
        "question",
    }
    return value if value in allowed else "factual"


def _coerce_confidence(value: str) -> str:
    allowed = {
        "source_backed",
        "strong",
        "moderate",
        "weak",
        "unsupported",
        "contradicted",
        "unknown",
    }
    return value if value in allowed else "unknown"


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _finding_id(prefix: str, normalized: str, ordinal: int) -> str:
    return f"F-{hashlib.sha1(f'{prefix}:{ordinal}:{normalized}'.encode('utf-8')).hexdigest()[:10]}"


def _short_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
