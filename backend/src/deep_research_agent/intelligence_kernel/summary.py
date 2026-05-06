from __future__ import annotations

from pathlib import Path

from .contracts import (
    ConfidenceCalibration,
    CritiqueFinding,
    KernelArtifactMetadata,
    KernelRunSummary,
    KernelWarning,
    ResearchBlueprint,
    ResearchClaim,
    ResearchPass,
    SourceUnit,
    VerificationTask,
    write_json,
)


def build_kernel_summary(
    blueprint: ResearchBlueprint,
    passes: list[ResearchPass],
    source_units: list[SourceUnit],
    evidence_count: int,
    claims: list[ResearchClaim],
    findings: list[CritiqueFinding],
    tasks: list[VerificationTask],
    calibrations: list[ConfidenceCalibration],
    registry: list[KernelArtifactMetadata],
) -> KernelRunSummary:
    final = next((c for c in calibrations if c.target_type == "report"), None)
    if final is None:
        final = ConfidenceCalibration(
            target_type="report",
            target_id=blueprint.thread_id,
            confidence_before=0.0,
            confidence_after=0.0,
            level="very_low",
            reasons=["Confidence calibration did not run."],
        )
    warnings = list(blueprint.operator_warnings)
    warnings.extend(
        KernelWarning(
            warning_id=f.finding_id,
            subsystem="critique",
            code=f.category,
            severity=f.severity,
            message=f.message,
            affected_artifacts=f.affected_artifacts,
            affected_sources=f.affected_sources,
            affected_claims=f.affected_claims,
            recommended_action=f.recommended_action,
        )
        for f in findings
    )
    next_actions = []
    if any(w.severity == "critical" for w in warnings):
        next_actions.append("Resolve critical warnings before using the report.")
    if any(w.severity == "high" for w in warnings):
        next_actions.append("Review high-severity critique findings manually.")
    if len([u for u in source_units if u.source_role == "primary_evidence"]) == 0:
        next_actions.append("Add primary or official sources.")
    if len([t for t in tasks if t.status in {"unsupported", "contradicted"}]):
        next_actions.append("Revise unsupported or contradicted claims.")
    if blueprint.intent.label in {
        "legal_policy_review",
        "medical_health_review",
        "financial_risk_review",
    }:
        next_actions.append("Require qualified human review for sensitive-domain conclusions.")
    if not next_actions:
        next_actions.append("Use the report with normal citation review.")
    return KernelRunSummary(
        thread_id=blueprint.thread_id,
        question=blueprint.question,
        intent=blueprint.intent,
        complexity=blueprint.complexity,
        passes_executed=[p.pass_type for p in passes if p.status == "completed"],
        passes_skipped=[p.pass_type for p in passes if p.status == "skipped"],
        critical_warnings=[w for w in warnings if w.severity == "critical"],
        high_warnings=[w for w in warnings if w.severity == "high"],
        source_count=len(source_units),
        evidence_unit_count=evidence_count,
        claim_count=len(claims),
        verified_claim_count=len([t for t in tasks if t.status == "verified"]),
        unsupported_claim_count=len([t for t in tasks if t.status == "unsupported"]),
        contradicted_claim_count=len([t for t in tasks if t.status == "contradicted"]),
        final_confidence=final,
        generated_artifacts=[
            item.path for item in registry if item.exists and item.producer == "intelligence_kernel"
        ],
        operator_next_actions=next_actions,
    )


def render_summary_markdown(summary: KernelRunSummary) -> str:
    return (
        "\n".join(
            [
                "# Kernel Summary",
                "",
                f"Question: {summary.question}",
                f"Intent: `{summary.intent.label}`",
                f"Complexity: `{summary.complexity.level}`",
                f"Final confidence: **{summary.final_confidence.level}** ({summary.final_confidence.confidence_after:.2f})",
                "",
                "## Counts",
                "",
                f"- Sources: `{summary.source_count}`",
                f"- Evidence units: `{summary.evidence_unit_count}`",
                f"- Claims: `{summary.claim_count}`",
                f"- Verified claims: `{summary.verified_claim_count}`",
                f"- Unsupported claims: `{summary.unsupported_claim_count}`",
                f"- Contradicted claims: `{summary.contradicted_claim_count}`",
                "",
                "## High Warnings",
                "",
                *([f"- `{w.code}`: {w.message}" for w in summary.high_warnings] or ["- None"]),
                "",
                "## Operator Next Actions",
                "",
                *[f"- {item}" for item in summary.operator_next_actions],
            ]
        )
        + "\n"
    )


def render_readiness_markdown(
    summary: KernelRunSummary,
    source_units: list[SourceUnit],
    claims: list[ResearchClaim],
    tasks: list[VerificationTask],
) -> str:
    usable = (
        "No"
        if summary.final_confidence.confidence_after < 0.4
        or summary.critical_warnings
        or summary.contradicted_claim_count
        else "Review required"
        if summary.high_warnings or summary.unsupported_claim_count
        else "Yes, with citation review"
    )
    strongest = [u for u in source_units if u.source_role == "primary_evidence"][:5]
    risky = [
        u for u in source_units if u.source_role in {"risky_source", "weak_reference", "duplicate"}
    ][:8]
    weak_claims = [c for c in claims if c.confidence_score < 0.45][:10]
    lines = [
        "# Research Readiness",
        "",
        f"Is the report usable? **{usable}**",
        f"Confidence: **{summary.final_confidence.level}** ({summary.final_confidence.confidence_after:.2f})",
        "",
        "## Biggest Risks",
        "",
        *(
            [f"- {w.message}" for w in [*summary.critical_warnings, *summary.high_warnings][:10]]
            or ["- No high or critical warnings."]
        ),
        "",
        "## Manual Checks Required",
        "",
        *[f"- {item}" for item in summary.operator_next_actions],
        "",
        "## Weak Claims",
        "",
        *([f"- {c.text[:180]}" for c in weak_claims] or ["- No low-confidence claims detected."]),
        "",
        "## Strongest Sources",
        "",
        *(
            [f"- {u.title or u.url} ({u.trust_level}, {u.source_role})" for u in strongest]
            or ["- No primary evidence sources detected."]
        ),
        "",
        "## Sources Not To Trust Blindly",
        "",
        *(
            [f"- {u.title or u.url} ({u.source_role})" for u in risky]
            or ["- No risky or weak sources detected."]
        ),
        "",
        "## Verification Snapshot",
        "",
        f"- Verified: `{len([t for t in tasks if t.status == 'verified'])}`",
        f"- Partially verified: `{len([t for t in tasks if t.status == 'partially_verified'])}`",
        f"- Unsupported: `{len([t for t in tasks if t.status == 'unsupported'])}`",
        f"- Contradicted: `{len([t for t in tasks if t.status == 'contradicted'])}`",
        "",
        "## Limitations",
        "",
        "- Verification is deterministic and uses local artifacts only.",
        "- This audit layer does not replace human review, especially for legal, medical, or financial domains.",
    ]
    return "\n".join(lines) + "\n"


def write_summary_artifacts(
    run_dir: Path,
    summary: KernelRunSummary,
    source_units: list[SourceUnit],
    claims: list[ResearchClaim],
    tasks: list[VerificationTask],
) -> list[str]:
    write_json(run_dir / "kernel_summary.json", summary)
    (run_dir / "kernel_summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    (run_dir / "research_readiness.md").write_text(
        render_readiness_markdown(summary, source_units, claims, tasks),
        encoding="utf-8",
    )
    return ["kernel_summary.json", "kernel_summary.md", "research_readiness.md"]
