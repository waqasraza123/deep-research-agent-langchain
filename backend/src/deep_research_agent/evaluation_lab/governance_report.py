from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    BenchmarkBaseline,
    BenchmarkCoverageSummary,
    GateTriageSummary,
    QualityGateProfile,
    QualityGateRunResult,
    WarningAudit,
    model_to_plain,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_to_plain(payload), indent=2, sort_keys=True) + "\n", "utf-8")


def render_gate_profile(profile: QualityGateProfile) -> str:
    lines = [
        f"# Quality Gate Profile: {profile.gate_id}",
        "",
        f"- Name: {profile.name}",
        f"- Enabled: {profile.enabled}",
        f"- Minimum pass rate: {profile.minimum_pass_rate:.3f}",
        f"- Minimum average score: {profile.minimum_average_score:.3f}",
        f"- Minimum case score: {profile.minimum_case_score}",
        f"- Compare against baseline: {profile.compare_against_baseline}",
        "",
        "## Selectors",
        "",
        f"- Cases: {', '.join(profile.case_ids) or 'all/derived'}",
        f"- Categories: {', '.join(c.value for c in profile.categories) or 'none'}",
        f"- Tags: {', '.join(profile.tags) or 'none'}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_warning_audit(audit: WarningAudit) -> str:
    lines = [
        "# Warning Audit",
        "",
        f"- Total warnings: {audit.total_warnings}",
        f"- Serious warnings: {audit.serious_warning_count}",
        f"- Budget exceeded: {audit.budget_exceeded}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(audit.warnings_by_category.items()))
    if audit.recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {item}" for item in audit.recommendations)
    return "\n".join(lines).rstrip() + "\n"


def render_coverage(summary: BenchmarkCoverageSummary) -> str:
    lines = [
        "# Benchmark Coverage",
        "",
        f"- Total cases: {summary.total_cases}",
        f"- Coverage score: {summary.coverage_score:.3f}",
        f"- Categories covered: {', '.join(summary.categories_covered) or 'none'}",
        f"- Categories missing: {', '.join(summary.categories_missing) or 'none'}",
        f"- Trap types covered: {', '.join(summary.trap_types_covered) or 'none'}",
        f"- Check types covered: {', '.join(summary.check_types_covered) or 'none'}",
    ]
    if summary.recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {item}" for item in summary.recommendations)
    return "\n".join(lines).rstrip() + "\n"


def render_triage(summary: GateTriageSummary) -> str:
    lines = ["# Gate Triage Summary", "", "## Highest Priority Failures", ""]
    lines.extend(f"- {item}" for item in summary.highest_priority_failures or ["None"])
    lines.extend(["", "## Likely Root Causes", ""])
    lines.extend(f"- {item}" for item in summary.likely_root_causes or ["None identified"])
    lines.extend(["", "## Suggested Engineering Tasks", ""])
    lines.extend(
        f"- {item}" for item in summary.suggested_engineering_tasks or ["No action required"]
    )
    if summary.cases_to_inspect_first:
        lines.extend(["", "## Cases To Inspect First", ""])
        lines.extend(f"- `{case_id}`" for case_id in summary.cases_to_inspect_first)
    return "\n".join(lines).rstrip() + "\n"


def render_quality_gate_run(
    result: QualityGateRunResult, profile: QualityGateProfile | None = None
) -> str:
    lines = [
        f"# Quality Gate Run {result.gate_run_id}",
        "",
        f"- Final status: {result.status.value}",
        f"- Gate profile: `{result.gate_id}`",
        f"- Benchmark run: `{result.benchmark_run_id}`",
        f"- Baseline: `{result.baseline_id or 'none'}`",
        f"- Pass rate: {result.pass_rate:.3f}",
        f"- Average score: {result.average_score:.3f}",
        f"- Minimum case score: {result.minimum_case_score:.3f}",
        f"- Cases: {result.passed_cases}/{result.total_cases} passed",
        "",
        "## Thresholds",
        "",
    ]
    for threshold in [*result.failed_thresholds, *result.passed_thresholds]:
        state = "PASS" if threshold.passed else "FAIL"
        lines.append(f"- {state} `{threshold.threshold_id}`: {threshold.message}")
    if not result.failed_thresholds and not result.passed_thresholds:
        lines.append("- No thresholds evaluated")
    lines.extend(["", "## Failed Cases", ""])
    failed_cases = sorted(
        {
            case_id
            for threshold in result.failed_thresholds
            for case_id in threshold.affected_cases
            if case_id
        }
    )
    lines.extend(f"- `{case_id}`" for case_id in failed_cases or ["None"])
    lines.extend(["", "## Regressions", ""])
    if result.regressions:
        lines.extend(f"- {finding.message}" for finding in result.regressions)
    else:
        lines.append("- None")
    lines.extend(["", "## Improvements", ""])
    if result.improvements:
        lines.extend(f"- {finding.message}" for finding in result.improvements)
    else:
        lines.append("- None")
    lines.extend(["", "## Warning Budget", ""])
    lines.append(
        f"- {result.warning_audit.total_warnings} warnings; "
        f"{result.warning_audit.serious_warning_count} serious; "
        f"budget exceeded: {result.warning_audit.budget_exceeded}"
    )
    lines.extend(["", "## Coverage", ""])
    lines.append(f"- Coverage score: {result.coverage_summary.coverage_score:.3f}")
    lines.extend(["", "## Triage Priorities", ""])
    lines.extend(
        f"- {item}" for item in result.triage_summary.highest_priority_failures or ["None"]
    )
    lines.extend(["", "## Recommended Actions", ""])
    lines.extend(f"- {item}" for item in result.recommended_actions or ["No action required"])
    if profile is not None:
        lines.extend(["", "## Profile Details", ""])
        lines.append(f"- {profile.description}")
    return "\n".join(lines).rstrip() + "\n"


def render_governance_summary(
    result: QualityGateRunResult, baseline: BenchmarkBaseline | None = None
) -> str:
    should_pass = "yes" if result.status.value == "passed" else "no"
    lines = [
        "# Governance Summary",
        "",
        f"- Should this branch pass the quality gate? {should_pass}",
        f"- Final status: {result.status.value}",
        f"- Gate: `{result.gate_id}`",
        f"- Baseline used: `{baseline.baseline_id if baseline else result.baseline_id or 'none'}`",
        f"- Benchmark run: `{result.benchmark_run_id}`",
        "",
        "## What Failed",
        "",
    ]
    if result.failed_thresholds:
        lines.extend(f"- {threshold.message}" for threshold in result.failed_thresholds)
    else:
        lines.append("- Nothing")
    lines.extend(["", "## What Improved", ""])
    if result.improvements:
        lines.extend(f"- {item.message}" for item in result.improvements)
    else:
        lines.append("- No improvements detected")
    lines.extend(["", "## What Is Risky", ""])
    risks = [item.message for item in result.regressions] + result.triage_summary.risky_regressions
    lines.extend(f"- {risk}" for risk in risks or ["No new risk identified"])
    lines.extend(["", "## Fix First", ""])
    lines.extend(
        f"- {task}"
        for task in result.triage_summary.suggested_engineering_tasks[:8] or ["No fix needed"]
    )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{artifact}`" for artifact in result.report_artifacts)
    return "\n".join(lines).rstrip() + "\n"
