from __future__ import annotations

import hashlib
import re

from deep_research_agent.evidence.claim_extractor import extract_values, split_sentences
from deep_research_agent.evidence.contracts import EvidenceLedger

from .contracts import HallucinationRisk, HallucinationRiskFinding, Severity
from .coverage import extract_entities

_ABSOLUTE_RE = re.compile(
    r"\b(always|never|guaranteed|guarantees|proven|proves|undeniably|definitely|"
    r"all|none|must|best|only)\b",
    re.IGNORECASE,
)
_RECOMMEND_RE = re.compile(r"\b(should|must|recommend|recommended|best|prefer|avoid)\b", re.I)
_STRONG_RE = re.compile(
    r"\b(proves|guarantees|always|never|undeniably|definitely|must|best|only)\b", re.I
)
_CITATION_RE = re.compile(r"\[(?:S\d+(?:,\s*S\d+)*)\]", re.I)


def assess_hallucination_risk(
    *,
    report_text: str,
    notes_text: str,
    source_texts: list[str],
    evidence_ledger: EvidenceLedger | None = None,
) -> HallucinationRisk:
    support_text = "\n".join([notes_text, *source_texts])
    support_lower = support_text.lower()
    source_entities = {entity.lower() for entity in extract_entities(support_text)}
    report_entities = extract_entities(report_text)
    source_values = {value.lower() for value in extract_values(support_text)}
    report_values = extract_values(report_text)
    findings: list[HallucinationRiskFinding] = []

    for value in report_values:
        if value.lower() in source_values:
            continue
        sentence = _sentence_containing(report_text, value)
        kind = "unsupported_date" if _looks_like_date(value) else "unsupported_number"
        findings.append(
            _finding(
                kind,
                "high",
                sentence or value,
                f"Value `{value}` appears in the report but not in sources, notes, or evidence.",
                "Remove the value, qualify it, or add a source that contains it.",
            )
        )

    for entity in report_entities:
        key = entity.lower()
        if key in source_entities or key in {"report", "summary", "sources", "limitations"}:
            continue
        if len(entity) <= 3 and entity.isupper():
            continue
        findings.append(
            _finding(
                "introduced_entity",
                "medium",
                entity,
                "Entity appears in the report but was not seen in sources or notes.",
                "Verify this entity against source material or remove it.",
            )
        )

    for sentence in split_sentences(report_text):
        has_citation = bool(_CITATION_RE.search(sentence))
        if _STRONG_RE.search(sentence) and not has_citation:
            findings.append(
                _finding(
                    "weakly_supported_strong_claim",
                    "high",
                    sentence,
                    "Strong wording appears without nearby citation support.",
                    "Add specific evidence or soften the wording.",
                )
            )
        if _RECOMMEND_RE.search(sentence) and not has_citation:
            findings.append(
                _finding(
                    "unsupported_recommendation",
                    "medium",
                    sentence,
                    "Recommendation language appears without direct evidence support.",
                    "Tie the recommendation to cited evidence and assumptions.",
                )
            )
        if _ABSOLUTE_RE.search(sentence) and not _absolute_supported(sentence, support_lower):
            findings.append(
                _finding(
                    "absolute_wording",
                    "medium",
                    sentence,
                    "Absolute wording is not mirrored by strong support text.",
                    "Replace absolutes with bounded, source-specific language.",
                )
            )

    if evidence_ledger is not None:
        contradicted = [
            claim
            for claim in evidence_ledger.claims
            if claim.origin != "source" and claim.contradiction_ids
        ]
        for claim in contradicted:
            findings.append(
                _finding(
                    "ignored_contradiction",
                    "critical",
                    claim.text,
                    "Evidence ledger marks this generated claim as contradicted.",
                    "Resolve the contradiction in the report or explicitly explain uncertainty.",
                )
            )
        for claim in evidence_ledger.claims:
            if claim.origin == "source":
                continue
            if claim.support_level in {"weak", "unsupported"} and _STRONG_RE.search(claim.text):
                findings.append(
                    _finding(
                        "weakly_supported_strong_claim",
                        "high",
                        claim.text,
                        f"Evidence ledger support level is `{claim.support_level}`.",
                        "Soften the claim or add stronger source support.",
                    )
                )

    findings = _dedupe_findings(findings)
    weighted = sum(_severity_weight(item.severity) for item in findings)
    risk_score = min(1.0, round(weighted / 10.0, 3))
    severity = _severity_from_risk(risk_score)
    confidence = "high" if evidence_ledger and source_texts else "medium" if source_texts else "low"
    return HallucinationRisk(
        risk_score=risk_score,
        severity=severity,
        findings=findings,
        checked_values={
            "report_values": report_values,
            "support_values": sorted(source_values)[:100],
            "report_entities": report_entities,
        },
        confidence=confidence,
    )


def _sentence_containing(text: str, needle: str) -> str:
    for sentence in split_sentences(text):
        if needle in sentence:
            return sentence
    return ""


def _looks_like_date(value: str) -> bool:
    pattern = r"\b(?:19|20)\d{2}\b|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    return bool(re.search(pattern, value, re.I))


def _absolute_supported(sentence: str, support_lower: str) -> bool:
    terms = [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{3,}\b", sentence)]
    if not terms:
        return False
    overlap = sum(1 for term in terms if term in support_lower) / len(set(terms))
    return overlap >= 0.7


def _finding(
    kind: str,
    severity: Severity,
    text: str,
    reason: str,
    suggested_fix: str,
) -> HallucinationRiskFinding:
    digest = hashlib.sha1(f"{kind}:{text}:{reason}".encode("utf-8")).hexdigest()[:10]
    return HallucinationRiskFinding(
        finding_id=f"HR-{digest}",
        kind=kind,  # type: ignore[arg-type]
        severity=severity,
        text=text,
        reason=reason,
        suggested_fix=suggested_fix,
        affected_artifacts=["report.md", "evidence_ledger.json"],
    )


def _dedupe_findings(
    findings: list[HallucinationRiskFinding],
) -> list[HallucinationRiskFinding]:
    seen: set[tuple[str, str]] = set()
    out: list[HallucinationRiskFinding] = []
    for finding in findings:
        key = (finding.kind, finding.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out


def _severity_weight(severity: Severity) -> float:
    return {
        "info": 0.2,
        "low": 0.5,
        "medium": 1.0,
        "high": 1.8,
        "critical": 2.5,
    }[severity]


def _severity_from_risk(risk_score: float) -> Severity:
    if risk_score >= 0.75:
        return "critical"
    if risk_score >= 0.5:
        return "high"
    if risk_score >= 0.25:
        return "medium"
    if risk_score > 0:
        return "low"
    return "info"
