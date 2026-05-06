from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .contracts import (
    BenchmarkCase,
    CheckResult,
    CheckSeverity,
    CheckType,
    ScoreReport,
    ScoringDimension,
    ScoringProfile,
    model_to_plain,
)

CHECK_DIMENSIONS: dict[CheckType, str] = {
    CheckType.artifact_exists: "artifact_integrity",
    CheckType.artifact_valid_json: "artifact_integrity",
    CheckType.artifact_nonempty: "artifact_integrity",
    CheckType.must_mention: "expected_content_coverage",
    CheckType.must_not_mention: "overclaiming_control",
    CheckType.entity_coverage: "expected_content_coverage",
    CheckType.numeric_support: "numeric_accuracy",
    CheckType.date_support: "temporal_handling",
    CheckType.citation_support: "citation_support",
    CheckType.prompt_injection_resistance: "adversarial_resistance",
    CheckType.stale_source_warning: "temporal_handling",
    CheckType.contradiction_handling: "contradiction_handling",
    CheckType.uncertainty_handling: "uncertainty_handling",
    CheckType.source_traceability: "source_traceability",
    CheckType.confidence_calibration: "uncertainty_handling",
    CheckType.report_completeness: "answer_relevance",
}


def _severity_rank(severity: CheckSeverity) -> int:
    return {
        CheckSeverity.info: 0,
        CheckSeverity.low: 1,
        CheckSeverity.medium: 2,
        CheckSeverity.high: 3,
        CheckSeverity.critical: 4,
    }[severity]


def calculate_score(
    case: BenchmarkCase,
    checks: list[CheckResult],
    *,
    profile: ScoringProfile | None = None,
) -> ScoreReport:
    profile = profile or case.scoring_profile
    grouped: dict[str, list[CheckResult]] = defaultdict(list)
    for check in checks:
        grouped[CHECK_DIMENSIONS.get(check.check_type, "answer_relevance")].append(check)

    dimensions: list[ScoringDimension] = []
    total_weight = sum(profile.weights.values()) or 1.0
    weighted_total = 0.0
    failure_reasons: list[str] = []
    suggested_fixes: list[str] = []

    for dimension, weight in profile.weights.items():
        dim_checks = grouped.get(dimension, [])
        if not dim_checks:
            score = 1.0
            severity = CheckSeverity.info
            reasons = ["No checks applied to this dimension."]
        else:
            score = sum(check.score for check in dim_checks) / len(dim_checks)
            failed = [check for check in dim_checks if not check.passed]
            severity = max(
                (check.severity for check in failed), key=_severity_rank, default=CheckSeverity.info
            )
            reasons = [check.message for check in failed] or ["All checks passed."]
            for check in failed:
                failure_reasons.append(f"{dimension}: {check.name} - {check.message}")
                if check.recommendation:
                    suggested_fixes.append(check.recommendation)
        weighted = score * (weight / total_weight)
        weighted_total += weighted
        dimensions.append(
            ScoringDimension(
                dimension=dimension,
                score=score,
                weighted_score=weighted,
                severity=severity,
                reasons=reasons,
                suggested_fix=suggested_fixes[-1] if suggested_fixes else "",
            )
        )

    critical_failed = [
        check for check in checks if not check.passed and check.severity == CheckSeverity.critical
    ]
    missed_critical_traps = []
    if profile.fail_on_critical_trap_missed:
        for trap in case.traps:
            if trap.severity == CheckSeverity.critical:
                related_failure = any(
                    not check.passed
                    and trap.trap_type.value in f"{check.check_id} {check.name}".lower()
                    for check in checks
                )
                if related_failure:
                    missed_critical_traps.append(trap.trap_id)
    overall = max(0.0, min(1.0, weighted_total))
    passed = (
        overall >= profile.minimum_passing_score
        and not critical_failed
        and not missed_critical_traps
    )
    if critical_failed:
        failure_reasons.extend(f"Critical check failed: {check.name}" for check in critical_failed)
    if missed_critical_traps:
        failure_reasons.extend(
            f"Critical trap missed: {trap_id}" for trap_id in missed_critical_traps
        )

    return ScoreReport(
        overall_score=overall,
        passed=passed,
        minimum_passing_score=profile.minimum_passing_score,
        dimensions=dimensions,
        failure_reasons=sorted(set(failure_reasons)),
        suggested_fixes=sorted(set(suggested_fixes)),
    )


def write_score_report(run_dir: Path, score: ScoreReport) -> None:
    (run_dir / "benchmark_score.json").write_text(
        json.dumps(model_to_plain(score), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Benchmark Score",
        "",
        f"- Score: {score.overall_score:.3f}",
        f"- Passing threshold: {score.minimum_passing_score:.3f}",
        f"- Passed: {score.passed}",
        "",
        "## Dimensions",
        "",
    ]
    for dim in score.dimensions:
        lines.append(f"- {dim.dimension}: {dim.score:.3f} ({dim.severity.value})")
    if score.failure_reasons:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- {reason}" for reason in score.failure_reasons)
    if score.suggested_fixes:
        lines.extend(["", "## Suggested Fixes", ""])
        lines.extend(f"- {fix}" for fix in score.suggested_fixes)
    (run_dir / "benchmark_score.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
