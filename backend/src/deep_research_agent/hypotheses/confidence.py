from __future__ import annotations

import hashlib
import re

from .contracts import (
    ConfidenceLevel,
    HypothesisConfidenceUpdate,
    HypothesisSet,
    HypothesisStatus,
    HypothesisTestResult,
    ResearchHypothesis,
)
from .generator import HypothesisBuildInput

_STRONG_LANGUAGE_RE = re.compile(
    r"\b(always|never|proves|guarantees|undeniably|definitely|best|must|only|all|none)\b",
    re.IGNORECASE,
)
_FRESHNESS_RE = re.compile(
    r"\b(current|currently|latest|recent|today|now|as of|this year|newest|202\d)\b",
    re.IGNORECASE,
)


def update_confidences(
    hypothesis_set: HypothesisSet,
    build_input: HypothesisBuildInput,
) -> HypothesisSet:
    results = {result.hypothesis_id: result for result in hypothesis_set.test_results}
    updates: list[HypothesisConfidenceUpdate] = []
    for hypothesis in hypothesis_set.hypotheses:
        result = results.get(hypothesis.hypothesis_id)
        if result is None:
            continue
        update = confidence_update_for_hypothesis(hypothesis, result, build_input)
        updates.append(update)
        hypothesis.confidence_update_id = update.update_id
    hypothesis_set.confidence_updates = updates
    return hypothesis_set


def confidence_update_for_hypothesis(
    hypothesis: ResearchHypothesis,
    result: HypothesisTestResult,
    build_input: HypothesisBuildInput,
) -> HypothesisConfidenceUpdate:
    sensitivity = _domain_sensitivity(hypothesis, build_input.question)
    prior = 0.30 if sensitivity != "normal" else 0.35
    score = prior
    factors: list[str] = []
    penalties: list[str] = []
    missing: list[str] = []

    support_sources = len(result.supporting_source_ids)
    opposing_sources = len(result.opposing_source_ids)
    if support_sources:
        delta = min(0.24, support_sources * 0.08)
        score += delta
        factors.append(f"{support_sources} supporting source(s).")
    else:
        score -= 0.08
        missing.append("No supporting source matched the hypothesis.")

    if opposing_sources:
        delta = min(0.30, opposing_sources * 0.10)
        score -= delta
        penalties.append(f"{opposing_sources} opposing source(s).")

    if result.primary_source_count:
        score += min(0.12, result.primary_source_count * 0.06)
        factors.append(f"{result.primary_source_count} primary-source signal(s).")
    else:
        missing.append("No primary-source signal was available.")

    if result.citation_ready_count:
        score += min(0.08, result.citation_ready_count * 0.04)
        factors.append(f"{result.citation_ready_count} citation-ready source(s).")

    if result.source_diversity >= 3:
        score += 0.06
        factors.append("Evidence spans at least three sources.")
    elif result.source_diversity <= 1:
        score -= 0.05
        penalties.append("Evidence has low source diversity.")

    if result.support_score >= 0.68:
        score += 0.08
        factors.append("Best evidence has strong deterministic match specificity.")
    elif result.support_score and result.support_score < 0.42:
        score -= 0.05
        penalties.append("Matched evidence is weak or generic.")

    if result.contradiction_ids:
        score -= min(0.24, len(result.contradiction_ids) * 0.08)
        penalties.append("Linked contradiction warnings remain unresolved.")

    if hypothesis.status == HypothesisStatus.CONTRADICTED:
        score -= 0.20
        penalties.append("Hypothesis is contradicted by available evidence.")
    elif hypothesis.status == HypothesisStatus.UNSUPPORTED:
        score -= 0.14
        penalties.append("Hypothesis is unsupported by deterministic evidence.")
    elif hypothesis.status == HypothesisStatus.NEEDS_MORE_EVIDENCE:
        score -= 0.12
        missing.append("Additional evidence is required before testing this hypothesis.")
    elif hypothesis.status == HypothesisStatus.PARTIALLY_SUPPORTED:
        score -= 0.02
        missing.append("Support is partial and needs corroboration.")

    if _STRONG_LANGUAGE_RE.search(hypothesis.text) and support_sources < 2:
        score -= 0.10
        penalties.append("Strong language has fewer than two supporting sources.")

    if _FRESHNESS_RE.search(hypothesis.text) and not _has_fresh_support(result):
        score -= 0.06
        missing.append("Freshness-sensitive hypothesis lacks current/recent source signal.")

    temporal_penalty, temporal_messages = _temporal_penalties(build_input, result, hypothesis)
    if temporal_penalty:
        score -= temporal_penalty
        penalties.extend(temporal_messages)

    quantitative_penalty, quantitative_messages = _quantitative_penalties(
        build_input,
        hypothesis,
    )
    if quantitative_penalty:
        score -= quantitative_penalty
        penalties.extend(quantitative_messages)

    safety_penalty, safety_messages = _source_safety_penalties(build_input, result)
    if safety_penalty:
        score -= safety_penalty
        penalties.extend(safety_messages)

    if sensitivity != "normal":
        score -= 0.08
        penalties.append(f"{sensitivity} domain requires conservative confidence.")

    if (
        _requires_primary_source(hypothesis, build_input.question)
        and not result.primary_source_count
    ):
        score -= 0.08
        missing.append("A primary/official source is required but was not matched.")

    posterior = round(max(0.0, min(1.0, score)), 3)
    level = _confidence_level(posterior)
    needs_review = (
        posterior < 0.70
        or hypothesis.status
        in {
            HypothesisStatus.CONTRADICTED,
            HypothesisStatus.UNSUPPORTED,
            HypothesisStatus.INCONCLUSIVE,
            HypothesisStatus.NEEDS_MORE_EVIDENCE,
        }
        or sensitivity != "normal"
    )
    return HypothesisConfidenceUpdate(
        update_id=_update_id(hypothesis.hypothesis_id),
        hypothesis_id=hypothesis.hypothesis_id,
        prior_score=prior,
        posterior_score=posterior,
        confidence_level=level,
        factors=factors,
        penalties=penalties,
        missing_evidence=missing,
        domain_sensitivity=sensitivity,
        needs_human_review=needs_review,
    )


def _confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.86:
        return ConfidenceLevel.VERY_HIGH
    if score >= 0.70:
        return ConfidenceLevel.HIGH
    if score >= 0.50:
        return ConfidenceLevel.MEDIUM
    if score >= 0.30:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def _domain_sensitivity(hypothesis: ResearchHypothesis, question: str) -> str:
    text = f"{question} {hypothesis.text}".lower()
    if re.search(r"\b(legal|law|policy|regulation|compliance|license)\b", text):
        return "legal_policy"
    if re.search(r"\b(financial|investment|medical|health|clinical)\b", text):
        return "high_stakes"
    if re.search(r"\b(market|vendor|pricing|forecast)\b", text):
        return "market"
    if re.search(r"\b(security|risk|incident|failure)\b", text):
        return "risk"
    return "normal"


def _requires_primary_source(hypothesis: ResearchHypothesis, question: str) -> bool:
    text = f"{question} {hypothesis.text}".lower()
    return bool(
        re.search(
            r"\b(legal|policy|regulation|official|production|security|benchmark|current|latest)\b",
            text,
        )
    )


def _has_fresh_support(result: HypothesisTestResult) -> bool:
    for evidence in result.supporting_evidence:
        if evidence.freshness_status in {"current", "recent"}:
            return True
    return False


def _temporal_penalties(
    build_input: HypothesisBuildInput,
    result: HypothesisTestResult,
    hypothesis: ResearchHypothesis,
) -> tuple[float, list[str]]:
    currentness = build_input.currentness_assessment or {}
    if not currentness:
        return 0.0, []
    penalties: list[str] = []
    score = 0.0
    freshness_required = bool(currentness.get("freshness_required")) or bool(
        _FRESHNESS_RE.search(hypothesis.text)
    )
    stale_sources = set(str(item) for item in currentness.get("stale_sources") or [])
    unknown_sources = set(str(item) for item in currentness.get("unknown_date_sources") or [])
    supporting_sources = set(result.supporting_source_ids)
    if stale_sources & supporting_sources:
        score += 0.08
        penalties.append("Supporting evidence includes stale source(s).")
    if freshness_required and unknown_sources & supporting_sources:
        score += 0.06
        penalties.append("Freshness-sensitive hypothesis relies on undated source(s).")
    status = str(currentness.get("status") or "")
    if freshness_required and status in {"stale", "possibly_stale", "unknown"}:
        score += 0.04
        penalties.append(f"Run currentness status is {status}.")
    return min(score, 0.18), penalties


def _quantitative_penalties(
    build_input: HypothesisBuildInput,
    hypothesis: ResearchHypothesis,
) -> tuple[float, list[str]]:
    profile = build_input.quantitative_profile or {}
    evidence = profile.get("evidence") if isinstance(profile.get("evidence"), dict) else {}
    warnings = profile.get("warnings") or (evidence or {}).get("warnings") or []
    comparisons = (evidence or {}).get("comparisons") or []
    numeric_claims = (evidence or {}).get("numeric_claims") or []
    has_numbers = bool(re.search(r"\d", hypothesis.text))
    consistency_failures = int(profile.get("consistency_failures") or 0)
    score = 0.0
    penalties: list[str] = []
    if has_numbers and warnings:
        score += 0.06
        penalties.append("Quantitative warnings exist for numeric evidence.")
    high_severity = any(
        isinstance(warning, dict)
        and str(warning.get("severity") or "").lower() in {"high", "critical", "error"}
        for warning in warnings
    )
    if high_severity:
        score += 0.04
        penalties.append("High-severity quantitative warning affects the run.")
    if consistency_failures:
        score += min(0.08, consistency_failures * 0.04)
        penalties.append("Quantitative consistency failures remain unresolved.")
    if has_numbers and any(
        isinstance(claim, dict) and claim.get("support_status") in {"unsupported", "warning"}
        for claim in numeric_claims
    ):
        score += 0.08
        penalties.append("Numeric claims include unsupported or warning status.")
    if any(isinstance(item, dict) and item.get("comparable") is False for item in comparisons):
        score += 0.06
        penalties.append("At least one quantitative comparison is not apples-to-apples.")
    return min(score, 0.16), penalties


def _source_safety_penalties(
    build_input: HypothesisBuildInput,
    result: HypothesisTestResult,
) -> tuple[float, list[str]]:
    safety = build_input.source_safety or {}
    assessments = safety.get("assessments") if isinstance(safety, dict) else []
    if not isinstance(assessments, list):
        return 0.0, []
    by_source = {
        str(item.get("source_id")): item
        for item in assessments
        if isinstance(item, dict) and item.get("source_id")
    }
    support_sources = set(result.supporting_source_ids)
    high_risk = [
        sid
        for sid in support_sources
        if str((by_source.get(sid) or {}).get("risk_score", {}).get("risk_level"))
        in {"high", "critical"}
    ]
    excluded = [
        sid
        for sid in support_sources
        if (by_source.get(sid) or {}).get("sanitized_content", {}).get("agent_context_allowed")
        is False
    ]
    penalties: list[str] = []
    score = 0.0
    if high_risk:
        score += 0.10
        penalties.append("Supporting evidence includes high-risk source-safety finding(s).")
    if excluded:
        score += 0.12
        penalties.append("Supporting evidence includes source(s) excluded from agent context.")
    return min(score, 0.18), penalties


def _update_id(hypothesis_id: str) -> str:
    digest = hashlib.sha1(hypothesis_id.encode("utf-8")).hexdigest()[:10]
    return f"HCU-{digest}"
