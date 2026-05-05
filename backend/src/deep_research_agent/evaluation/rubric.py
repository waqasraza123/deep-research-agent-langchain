from __future__ import annotations

import re
from typing import Any

from deep_research_agent.evidence.contracts import EvidenceLedger

from .contracts import (
    BalanceAssessment,
    CitationQualityAssessment,
    CoverageGap,
    CriterionScore,
    EvaluationCriterion,
    EvaluationRubric,
    FreshnessAssessment,
    HallucinationRisk,
    Severity,
)

CRITERION_KEYS = (
    "question_answered",
    "source_coverage",
    "source_quality",
    "citation_support",
    "freshness_handling",
    "uncertainty_handling",
    "contradiction_handling",
    "balance_and_counterarguments",
    "specificity",
    "actionability",
    "safety_and_overclaiming",
    "artifact_completeness",
)


def default_rubric() -> EvaluationRubric:
    descriptions = {
        "question_answered": "Does the report directly answer the user's question?",
        "source_coverage": "Does the run use enough relevant sources and cover fetched material?",
        "source_quality": "Are sources authoritative, diverse, and usable?",
        "citation_support": "Are generated claims supported by citations and evidence?",
        "freshness_handling": "Are time-sensitive claims dated and current enough?",
        "uncertainty_handling": "Does the report explain limits and uncertainty?",
        "contradiction_handling": "Are contradictions acknowledged and resolved?",
        "balance_and_counterarguments": "Are tradeoffs and counterarguments included when needed?",
        "specificity": "Is the report concrete rather than vague?",
        "actionability": "Are recommendations or next steps usable and qualified?",
        "safety_and_overclaiming": "Does the report avoid unsupported absolutes and unsafe claims?",
        "artifact_completeness": "Are expected artifacts present and usable?",
    }
    weights = {
        "question_answered": 1.2,
        "source_coverage": 1.0,
        "source_quality": 0.9,
        "citation_support": 1.3,
        "freshness_handling": 1.0,
        "uncertainty_handling": 0.8,
        "contradiction_handling": 1.0,
        "balance_and_counterarguments": 0.9,
        "specificity": 0.7,
        "actionability": 0.6,
        "safety_and_overclaiming": 1.1,
        "artifact_completeness": 0.8,
    }
    return EvaluationRubric(
        criteria=[
            EvaluationCriterion(
                key=key,
                name=key.replace("_", " ").title(),
                description=descriptions[key],
                weight=weights[key],
                minimum_score=0.65,
            )
            for key in CRITERION_KEYS
        ]
    )


def score_rubric(
    *,
    question: str,
    report_text: str,
    notes_text: str,
    sources: list[dict[str, Any]],
    artifacts_present: set[str],
    coverage_gaps: list[CoverageGap],
    hallucination_risk: HallucinationRisk,
    balance: BalanceAssessment,
    freshness: FreshnessAssessment,
    evidence_ledger: EvidenceLedger | None = None,
) -> tuple[EvaluationRubric, list[CriterionScore], CitationQualityAssessment, float]:
    rubric = default_rubric()
    citation_quality = assess_citation_quality(
        report_text=report_text,
        sources=sources,
        evidence_ledger=evidence_ledger,
    )
    scores = [
        _question_answered(question, report_text, coverage_gaps),
        _source_coverage(sources, coverage_gaps),
        _source_quality(sources),
        _citation_support(citation_quality),
        _from_assessment(
            "freshness_handling",
            freshness.score,
            freshness.severity,
            freshness.reasons,
        ),
        _uncertainty_handling(report_text, evidence_ledger),
        _contradiction_handling(report_text, evidence_ledger),
        _from_assessment(
            "balance_and_counterarguments",
            balance.score,
            balance.severity,
            balance.reasons,
        ),
        _specificity(report_text),
        _actionability(report_text, evidence_ledger),
        _safety(hallucination_risk),
        _artifact_completeness(artifacts_present),
    ]
    weights = {criterion.key: criterion.weight for criterion in rubric.criteria}
    total_weight = sum(weights.get(score.criterion_key, 1.0) for score in scores)
    overall = sum(score.score * weights.get(score.criterion_key, 1.0) for score in scores)
    overall_score = round(overall / total_weight, 3) if total_weight else 0.0
    return rubric, scores, citation_quality, overall_score


def assess_citation_quality(
    *,
    report_text: str,
    sources: list[dict[str, Any]],
    evidence_ledger: EvidenceLedger | None = None,
) -> CitationQualityAssessment:
    cited = set(re.findall(r"\[(S\d+)\]", report_text, flags=re.I))
    total_sources = len([source for source in sources if source.get("ok", True) is not False])
    unsupported = 0
    weak = 0
    if evidence_ledger is not None:
        unsupported = len(evidence_ledger.unsupported_claims)
        weak = len(
            [
                claim
                for claim in evidence_ledger.claims
                if claim.origin != "source" and claim.support_level in {"weak", "unsupported"}
            ]
        )

    score = 0.25
    if total_sources:
        score += min(0.3, len(cited) / total_sources * 0.3)
    if evidence_ledger is not None and evidence_ledger.coverage.generated_claims:
        supported_ratio = (
            evidence_ledger.coverage.supported_claims
            + 0.5 * evidence_ledger.coverage.partially_supported_claims
        ) / evidence_ledger.coverage.generated_claims
        score += min(0.35, supported_ratio * 0.35)
        score -= min(0.25, unsupported * 0.05)
    elif cited:
        score += 0.15
    else:
        score -= 0.15
    score = round(max(0.0, min(1.0, score)), 3)
    reasons: list[str] = []
    if not cited:
        reasons.append("Report has no explicit [Sx] citation markers.")
    if unsupported:
        reasons.append(f"{unsupported} unsupported claim(s) found in evidence ledger.")
    if weak:
        reasons.append(f"{weak} generated claim(s) have weak or unsupported evidence.")
    if total_sources and len(cited) < total_sources:
        reasons.append("Not all usable sources are cited in the report.")
    return CitationQualityAssessment(
        score=score,
        severity=_severity_for_score(score),
        cited_source_count=len(cited),
        total_source_count=total_sources,
        unsupported_claim_count=unsupported,
        weak_claim_count=weak,
        reasons=reasons or ["Citation support appears adequate for deterministic checks."],
    )


def _question_answered(
    question: str, report_text: str, coverage_gaps: list[CoverageGap]
) -> CriterionScore:
    terms = _terms(question)
    report_terms = set(_terms(report_text))
    overlap = len(set(terms) & report_terms) / max(len(set(terms)), 1)
    gap_penalty = sum(0.12 for gap in coverage_gaps if gap.kind == "unanswered_subquestion")
    score = max(0.0, min(1.0, 0.25 + overlap * 0.75 - gap_penalty))
    reasons = [f"Question/report term overlap is {overlap:.2f}."]
    if gap_penalty:
        reasons.append("One or more subquestions appear unanswered.")
    return _criterion("question_answered", score, reasons, "Answer each subquestion directly.")


def _source_coverage(
    sources: list[dict[str, Any]], coverage_gaps: list[CoverageGap]
) -> CriterionScore:
    usable = len([source for source in sources if source.get("ok", True) is not False])
    score = min(0.85, usable * 0.25)
    if usable >= 3:
        score = 0.9
    penalty = sum(
        0.12
        for gap in coverage_gaps
        if gap.kind in {"unused_fetched_url", "missing_primary_source", "missing_opposing_view"}
    )
    score = max(0.0, score - penalty)
    reasons = [f"{usable} usable source(s) detected."]
    if penalty:
        reasons.append("Coverage gaps reduced the source coverage score.")
    return _criterion("source_coverage", score, reasons, "Use and cite relevant fetched sources.")


def _source_quality(sources: list[dict[str, Any]]) -> CriterionScore:
    usable = [source for source in sources if source.get("ok", True) is not False]
    if not usable:
        return _criterion(
            "source_quality",
            0.0,
            ["No usable sources detected."],
            "Fetch usable sources.",
        )
    quality_values: list[float] = []
    for source in usable:
        raw = source.get("final_quality_score")
        if raw is None and isinstance(source.get("quality_score"), dict):
            raw = source["quality_score"].get("final_quality_score")
        try:
            quality_values.append(float(raw if raw is not None else 0.5))
        except Exception:
            quality_values.append(0.5)
    avg = sum(quality_values) / len(quality_values)
    score = round(max(0.0, min(1.0, avg)), 3)
    return _criterion(
        "source_quality",
        score,
        [f"Average source quality estimate is {score:.2f}."],
        "Prefer authoritative, dated, primary sources.",
    )


def _citation_support(citation_quality: CitationQualityAssessment) -> CriterionScore:
    return _criterion(
        "citation_support",
        citation_quality.score,
        citation_quality.reasons,
        "Cite every material claim with nearby source markers.",
        ["report.md", "evidence_ledger.json"],
    )


def _uncertainty_handling(
    report_text: str, evidence_ledger: EvidenceLedger | None
) -> CriterionScore:
    lower = report_text.lower()
    has_uncertainty = any(
        term in lower for term in ("uncertain", "limitation", "caveat", "depends", "not enough")
    )
    score = 0.8 if has_uncertainty else 0.45
    reasons = (
        ["Report includes uncertainty or limitation language."]
        if has_uncertainty
        else ["Report lacks explicit uncertainty or limitation language."]
    )
    if evidence_ledger and evidence_ledger.unsupported_claims and not has_uncertainty:
        score -= 0.15
        reasons.append("Unsupported claims require clearer uncertainty handling.")
    return _criterion(
        "uncertainty_handling",
        score,
        reasons,
        "Add caveats, confidence boundaries, and unresolved evidence gaps.",
    )


def _contradiction_handling(
    report_text: str, evidence_ledger: EvidenceLedger | None
) -> CriterionScore:
    if evidence_ledger is None:
        return _criterion(
            "contradiction_handling",
            0.45,
            ["No evidence ledger was available to verify contradictions."],
            "Build evidence artifacts and review contradiction warnings.",
        )
    if not evidence_ledger.contradictions:
        return _criterion(
            "contradiction_handling",
            0.9,
            ["No contradiction groups detected by evidence ledger."],
            "Continue checking conflicting claims against sources.",
        )
    acknowledges = any(
        term in report_text.lower()
        for term in ("contradict", "conflict", "mixed evidence", "disagree", "uncertain")
    )
    score = 0.6 if acknowledges else 0.25
    reasons = [f"{len(evidence_ledger.contradictions)} contradiction group(s) detected."]
    if not acknowledges:
        reasons.append("Report does not appear to acknowledge contradictions.")
    return _criterion(
        "contradiction_handling",
        score,
        reasons,
        "Explain contradictions and how they affect the conclusion.",
    )


def _specificity(report_text: str) -> CriterionScore:
    values = re.findall(r"\b(?:20\d{2}|\d+(?:\.\d+)?%?|\$?\d+(?:,\d{3})*)\b", report_text)
    citations = re.findall(r"\[(S\d+)\]", report_text, flags=re.I)
    score = min(1.0, 0.35 + min(0.25, len(values) * 0.03) + min(0.3, len(citations) * 0.06))
    if len(report_text.split()) >= 250:
        score += 0.1
    score = min(1.0, score)
    reasons = [f"Detected {len(values)} concrete value(s) and {len(citations)} citation marker(s)."]
    return _criterion("specificity", score, reasons, "Add concrete sourced facts and examples.")


def _actionability(report_text: str, evidence_ledger: EvidenceLedger | None) -> CriterionScore:
    lower = report_text.lower()
    has_action = any(
        term in lower for term in ("recommend", "should", "next step", "use ", "avoid")
    )
    if not has_action:
        return _criterion(
            "actionability",
            0.55,
            ["No clear recommendation or action path detected."],
            "Add qualified next steps when the question asks for guidance.",
        )
    unsupported_recommendations = 0
    if evidence_ledger is not None:
        unsupported_recommendations = len(
            [
                claim
                for claim in evidence_ledger.claims
                if claim.claim_type == "recommendation"
                and claim.support_level in {"weak", "unsupported"}
            ]
        )
    score = max(0.2, 0.8 - unsupported_recommendations * 0.15)
    reasons = ["Actionable language detected."]
    if unsupported_recommendations:
        reasons.append(
            f"{unsupported_recommendations} recommendation claim(s) are weakly supported."
        )
    return _criterion(
        "actionability",
        score,
        reasons,
        "Tie recommendations to evidence and assumptions.",
    )


def _safety(hallucination_risk: HallucinationRisk) -> CriterionScore:
    score = round(1.0 - hallucination_risk.risk_score, 3)
    reasons = [f"Hallucination risk score is {hallucination_risk.risk_score:.3f}."]
    reasons.extend(finding.reason for finding in hallucination_risk.findings[:5])
    return _criterion(
        "safety_and_overclaiming",
        score,
        reasons,
        "Remove unsupported values/entities and soften overclaims.",
    )


def _artifact_completeness(artifacts_present: set[str]) -> CriterionScore:
    expected = {
        "plan.md",
        "notes.md",
        "sources.json",
        "report.md",
        "evidence_ledger.json",
        "evidence_coverage.json",
    }
    present = len(expected & artifacts_present)
    score = present / len(expected)
    missing = sorted(expected - artifacts_present)
    reasons = [f"{present}/{len(expected)} core artifacts are present."]
    if missing:
        reasons.append("Missing: " + ", ".join(missing))
    return _criterion(
        "artifact_completeness",
        score,
        reasons,
        "Generate the missing core artifacts.",
        sorted(expected),
    )


def _from_assessment(
    key: str, score: float, severity: Severity, reasons: list[str]
) -> CriterionScore:
    return CriterionScore(
        criterion_key=key,
        score=round(max(0.0, min(1.0, score)), 3),
        severity=severity,
        reasons=reasons,
        suggested_fix="Review and address the listed assessment reasons.",
        affected_artifacts=["report.md", "sources.json"],
    )


def _criterion(
    key: str,
    score: float,
    reasons: list[str],
    suggested_fix: str,
    affected_artifacts: list[str] | None = None,
) -> CriterionScore:
    clean = round(max(0.0, min(1.0, score)), 3)
    return CriterionScore(
        criterion_key=key,
        score=clean,
        severity=_severity_for_score(clean),
        reasons=reasons,
        suggested_fix=suggested_fix,
        affected_artifacts=affected_artifacts or ["report.md"],
    )


def _terms(text: str) -> list[str]:
    stop = {"about", "after", "also", "and", "are", "for", "from", "how", "the", "what", "with"}
    return [
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b", text)
        if token.lower() not in stop
    ]


def _severity_for_score(score: float) -> Severity:
    if score < 0.25:
        return "critical"
    if score < 0.45:
        return "high"
    if score < 0.65:
        return "medium"
    if score < 0.8:
        return "low"
    return "info"
