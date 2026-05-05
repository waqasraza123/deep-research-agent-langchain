from __future__ import annotations

from typing import Any

from .contracts import (
    PromptInjectionFinding,
    RecommendedAction,
    RiskLevel,
    SourcePoisoningFinding,
    SourceRiskScore,
)

LEVEL_POINTS: dict[RiskLevel, int] = {
    "none": 0,
    "low": 8,
    "medium": 18,
    "high": 32,
    "critical": 55,
}


def risk_level_from_score(score: float, *, has_critical: bool = False) -> RiskLevel:
    if has_critical or score >= 85:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def action_for_risk(
    risk_level: RiskLevel,
    *,
    has_critical_injection: bool = False,
    has_high_official_spoof: bool = False,
) -> RecommendedAction:
    if risk_level == "critical" or has_critical_injection:
        return "exclude_from_agent_context"
    if risk_level == "high":
        return "quote_only" if not has_high_official_spoof else "require_human_review"
    if risk_level == "medium":
        return "allow_with_warning"
    if risk_level == "low":
        return "allow_with_warning"
    return "allow"


def score_source_risk(
    *,
    source_id: str,
    url: str = "",
    prompt_injection_findings: list[PromptInjectionFinding],
    source_poisoning_findings: list[SourcePoisoningFinding],
    metadata: dict[str, Any] | None = None,
) -> SourceRiskScore:
    metadata = metadata or {}
    reasons: list[str] = []

    injection_score = min(
        72.0, sum(LEVEL_POINTS[finding.risk_level] for finding in prompt_injection_findings)
    )
    if prompt_injection_findings:
        categories = sorted({finding.category for finding in prompt_injection_findings})
        reasons.append(
            f"Prompt-injection indicators detected: {', '.join(categories)} "
            f"({len(prompt_injection_findings)} finding(s))."
        )

    poisoning_score = min(
        48.0, sum(LEVEL_POINTS[finding.risk_level] for finding in source_poisoning_findings)
    )
    if source_poisoning_findings:
        categories = sorted({finding.category for finding in source_poisoning_findings})
        reasons.append(
            f"Source-poisoning indicators detected: {', '.join(categories)} "
            f"({len(source_poisoning_findings)} finding(s))."
        )

    metadata_score = 0.0
    if metadata.get("truncated"):
        metadata_score += 5
        reasons.append("Source extraction was truncated.")
    if metadata.get("strategy") == "jina":
        metadata_score += 5
        reasons.append("Source required fallback extraction; verify extraction fidelity.")
    if metadata.get("status_code") and int(metadata.get("status_code") or 0) >= 400:
        metadata_score += 12
        reasons.append("Source fetch status was not successful.")
    if metadata.get("skip_reason"):
        metadata_score += 8
        reasons.append(f"Source had skip reason: {metadata.get('skip_reason')}.")

    credibility_score = 0.0
    final_quality = _bounded_float(metadata.get("final_quality_score"))
    if final_quality is None and isinstance(metadata.get("quality_score"), dict):
        final_quality = _bounded_float(metadata["quality_score"].get("final_quality_score"))
    if final_quality is not None and final_quality < 0.25:
        credibility_score += 12
        reasons.append(f"Source quality score is low ({final_quality:.2f}).")
    citation = metadata.get("citation_readiness_score") or metadata.get("citation_readiness")
    if isinstance(citation, dict):
        citation = citation.get("score")
    citation_score = _bounded_float(citation)
    if citation_score is not None and citation_score < 0.30:
        credibility_score += 8
        reasons.append(f"Citation readiness is weak ({citation_score:.2f}).")

    total = min(100.0, injection_score + poisoning_score + metadata_score + credibility_score)
    has_critical_injection = any(f.risk_level == "critical" for f in prompt_injection_findings)
    has_critical = has_critical_injection or any(
        f.risk_level == "critical" for f in source_poisoning_findings
    )
    risk_level = risk_level_from_score(total, has_critical=has_critical)
    if not reasons:
        reasons.append("No deterministic source-safety indicators were detected.")
    action = action_for_risk(
        risk_level,
        has_critical_injection=has_critical_injection,
        has_high_official_spoof=any(
            f.category == "fake_official_source_claim" and f.risk_level == "high"
            for f in source_poisoning_findings
        ),
    )
    return SourceRiskScore(
        source_id=source_id,
        url=url,
        numeric_score=total,
        risk_level=risk_level,
        recommended_action=action,
        prompt_injection_score=injection_score,
        poisoning_score=poisoning_score,
        metadata_score=metadata_score,
        credibility_score=credibility_score,
        reasons=reasons,
    )


def _bounded_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, f))
