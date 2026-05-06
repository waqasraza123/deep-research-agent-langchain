from __future__ import annotations

from pathlib import Path

from .contracts import (
    ConfidenceCalibration,
    CritiqueFinding,
    ResearchBlueprint,
    ResearchClaim,
    SourceUnit,
    VerificationTask,
    clamp_score,
    write_json,
)


def confidence_level(score: float) -> str:
    if score < 0.2:
        return "very_low"
    if score < 0.4:
        return "low"
    if score < 0.68:
        return "medium"
    if score < 0.86:
        return "high"
    return "very_high"


def calibrate_confidence(
    blueprint: ResearchBlueprint,
    source_units: list[SourceUnit],
    claims: list[ResearchClaim],
    findings: list[CritiqueFinding],
    tasks: list[VerificationTask],
) -> list[ConfidenceCalibration]:
    calibrations: list[ConfidenceCalibration] = []
    task_by_claim = {task.claim_id: task for task in tasks}
    for claim in claims:
        base = claim.confidence_score or 0.45
        reasons = []
        penalties = []
        boosts = []
        task = task_by_claim.get(claim.claim_id)
        if task:
            base = (base + task.confidence_after) / 2
            reasons.append(f"Verification status: {task.status}.")
            if task.status == "verified":
                base += 0.18
                boosts.append("Verified by local evidence.")
            elif task.status == "partially_verified":
                base += 0.06
                boosts.append("Partially verified by local evidence.")
            elif task.status in {"unsupported", "contradicted"}:
                base -= 0.24
                penalties.append(f"Verification result is {task.status}.")
        if claim.strength == "absolute":
            base -= 0.12
            penalties.append("Absolute language.")
        if claim.claim_type in {"legal_policy", "medical_health", "financial"}:
            base = min(base, 0.58)
            penalties.append("Sensitive-domain claim confidence is capped.")
        if claim.numbers and task and task.status != "verified":
            base -= 0.08
            penalties.append("Numeric claim was not fully verified.")
        if claim.supporting_evidence_ids:
            base += min(0.12, 0.03 * len(claim.supporting_evidence_ids))
            boosts.append("Linked supporting evidence.")
        score = clamp_score(base)
        calibrations.append(
            ConfidenceCalibration(
                target_type="claim",
                target_id=claim.claim_id,
                confidence_before=claim.confidence_score,
                confidence_after=score,
                level=confidence_level(score),  # type: ignore[arg-type]
                reasons=reasons or ["Deterministic claim scoring."],
                penalties=penalties,
                boosts=boosts,
                warnings=claim.warnings,
            )
        )
        claim.confidence_score = score
    source_diversity = len(
        {u.domain for u in source_units if u.source_role not in {"duplicate", "risky_source"}}
    )
    score = 0.62
    penalties = []
    boosts = []
    if not source_units:
        score -= 0.35
        penalties.append("No source units.")
    if source_diversity <= 1:
        score -= 0.16
        penalties.append("Low source diversity.")
    else:
        score += min(0.12, source_diversity * 0.025)
        boosts.append("Multiple source domains.")
    primary = len([u for u in source_units if u.source_role == "primary_evidence"])
    if primary:
        score += min(0.16, primary * 0.04)
        boosts.append("Primary evidence sources detected.")
    else:
        score -= 0.1
        penalties.append("No primary evidence sources.")
    high_findings = len([f for f in findings if f.severity in {"high", "critical"}])
    medium_findings = len([f for f in findings if f.severity == "medium"])
    score -= min(0.32, high_findings * 0.08 + medium_findings * 0.025)
    if high_findings:
        penalties.append("High-severity critique findings.")
    verified = len([t for t in tasks if t.status == "verified"])
    unsupported = len([t for t in tasks if t.status in {"unsupported", "contradicted"}])
    if verified:
        score += min(0.18, verified * 0.035)
        boosts.append("Claims verified by evidence.")
    if unsupported:
        score -= min(0.22, unsupported * 0.055)
        penalties.append("Unsupported or contradicted verification tasks.")
    if blueprint.intent.label in {
        "legal_policy_review",
        "medical_health_review",
        "financial_risk_review",
    }:
        score = min(score, 0.56)
        penalties.append("Sensitive-domain report confidence is capped.")
    final_score = clamp_score(score)
    calibrations.append(
        ConfidenceCalibration(
            target_type="report",
            target_id=blueprint.thread_id,
            confidence_before=0.62,
            confidence_after=final_score,
            level=confidence_level(final_score),  # type: ignore[arg-type]
            reasons=[
                "Deterministic confidence calibration from sources, critique, and verification."
            ],
            penalties=penalties,
            boosts=boosts,
            warnings=blueprint.operator_warnings,
        )
    )
    return calibrations


def render_confidence_markdown(calibrations: list[ConfidenceCalibration]) -> str:
    report = next(
        (c for c in calibrations if c.target_type == "report"),
        calibrations[-1] if calibrations else None,
    )
    lines = ["# Confidence Calibration", ""]
    if report:
        lines.extend(
            [
                f"Final confidence: **{report.level}** ({report.confidence_after:.2f})",
                "",
                "## Main Penalties",
                "",
                *([f"- {item}" for item in report.penalties] or ["- None"]),
                "",
                "## Main Boosts",
                "",
                *([f"- {item}" for item in report.boosts] or ["- None"]),
                "",
            ]
        )
    dist: dict[str, int] = {}
    for cal in calibrations:
        if cal.target_type == "claim":
            dist[cal.level] = dist.get(cal.level, 0) + 1
    lines.extend(["## Claim Confidence Distribution", ""])
    lines.extend(f"- {level}: `{count}`" for level, count in sorted(dist.items()))
    if not dist:
        lines.append("- No claims extracted.")
    lines.extend(
        [
            "",
            "## What Would Improve Confidence",
            "",
            "- Add primary sources.",
            "- Add citation-ready evidence for unsupported claims.",
            "- Resolve high-severity critique findings.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_confidence_artifacts(
    run_dir: Path, calibrations: list[ConfidenceCalibration]
) -> list[str]:
    write_json(run_dir / "confidence_calibration.json", {"calibrations": calibrations})
    (run_dir / "confidence_calibration.md").write_text(
        render_confidence_markdown(calibrations), encoding="utf-8"
    )
    return ["confidence_calibration.json", "confidence_calibration.md"]
