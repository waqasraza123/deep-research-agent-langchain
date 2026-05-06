from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .contracts import (
    BenchmarkCaseResult,
    BenchmarkRunResult,
    EvaluationLabSummary,
    RegressionComparison,
    model_to_plain,
    now_iso_utc,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_to_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_case_result(result: BenchmarkCaseResult) -> str:
    lines = [
        f"# {result.title}",
        "",
        f"- Case: `{result.case_id}`",
        f"- Status: {result.status}",
        f"- Score: {result.score:.3f}",
        f"- Passed: {result.passed}",
        f"- Thread: `{result.thread_id}`",
        f"- Run directory: `{result.run_dir}`",
        "",
        "## Failed Checks",
        "",
    ]
    failed = [check for check in result.check_results if not check.passed]
    if failed:
        for check in failed:
            lines.append(
                f"- {check.severity.value}: `{check.check_id}` {check.name} - {check.message}"
            )
            if check.recommendation:
                lines.append(f"  Suggested fix: {check.recommendation}")
    else:
        lines.append("- None")
    if result.missed_traps:
        lines.extend(["", "## Missed Traps", ""])
        lines.extend(f"- {trap}" for trap in result.missed_traps)
    if result.artifact_paths:
        lines.extend(["", "## Artifacts", ""])
        lines.extend(f"- `{artifact}`" for artifact in result.artifact_paths)
    return "\n".join(lines).rstrip() + "\n"


def write_case_result(run_dir: Path, result: BenchmarkCaseResult) -> None:
    write_json(run_dir / "case_result.json", result)
    (run_dir / "case_result.md").write_text(render_case_result(result), encoding="utf-8")


def build_summary(result: BenchmarkRunResult) -> EvaluationLabSummary:
    total = result.total_cases or len(result.case_results)
    pass_rate = (result.passed_cases / total) if total else 0.0
    failed = [case for case in result.case_results if not case.passed]
    failure_counts = Counter()
    for case in failed:
        for check in case.check_results:
            if not check.passed:
                failure_counts[check.check_type.value] += 1
    critical = [
        f"{case.case_id}: {check.name}"
        for case in failed
        for check in case.check_results
        if not check.passed and check.severity.value == "critical"
    ]
    fixes = []
    for case in failed:
        for check in case.check_results:
            if check.recommendation:
                fixes.append(check.recommendation)
    return EvaluationLabSummary(
        run_id=result.run_id,
        total_cases=total,
        pass_rate=pass_rate,
        average_score=result.average_score,
        highest_risk_failures=critical[:10],
        most_common_failures=[f"{name}: {count}" for name, count in failure_counts.most_common(10)],
        missed_critical_traps=[trap for case in failed for trap in case.missed_traps],
        artifact_quality_summary=f"{result.passed_cases}/{total} benchmark cases passed.",
        recommended_fixes=sorted(set(fixes))[:12],
        generated_at=now_iso_utc(),
    )


def render_run_result(result: BenchmarkRunResult) -> str:
    lines = [
        f"# Evaluation Lab Run {result.run_id}",
        "",
        f"- Status: {result.status}",
        f"- Total cases: {result.total_cases}",
        f"- Passed: {result.passed_cases}",
        f"- Failed: {result.failed_cases}",
        f"- Errored: {result.errored_cases}",
        f"- Average score: {result.average_score:.3f}",
        "",
        "## Cases",
        "",
    ]
    for case in result.case_results:
        state = "PASS" if case.passed else "FAIL"
        lines.append(f"- {state} `{case.case_id}` {case.score:.3f} - {case.title}")
    failed = [case for case in result.case_results if not case.passed]
    if failed:
        lines.extend(["", "## Recommended Next Engineering Work", ""])
        for case in failed:
            for reason in case.failure_reasons[:3]:
                lines.append(f"- `{case.case_id}`: {reason}")
    return "\n".join(lines).rstrip() + "\n"


def write_run_reports(run_dir: Path, result: BenchmarkRunResult) -> EvaluationLabSummary:
    write_json(run_dir / "benchmark_run.json", result)
    (run_dir / "benchmark_run.md").write_text(render_run_result(result), encoding="utf-8")
    summary = build_summary(result)
    write_json(run_dir / "evaluation_lab_summary.json", summary)
    (run_dir / "evaluation_lab_summary.md").write_text(summary.to_markdown(), encoding="utf-8")
    return summary


def write_comparison(path: Path, comparison: RegressionComparison) -> None:
    write_json(path / "regression_comparison.json", comparison)
    lines = [
        "# Regression Comparison",
        "",
        f"- Baseline: `{comparison.baseline_run_id}`",
        f"- Current: `{comparison.current_run_id}`",
        f"- Score delta: {comparison.score_delta:.3f}",
        f"- Pass-rate delta: {comparison.pass_rate_delta:.3f}",
        "",
        "## Newly Failed",
        "",
        *(f"- `{case_id}`" for case_id in comparison.newly_failed_cases),
        "",
        "## Newly Passed",
        "",
        *(f"- `{case_id}`" for case_id in comparison.newly_passed_cases),
        "",
        comparison.summary,
    ]
    (path / "regression_comparison.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
