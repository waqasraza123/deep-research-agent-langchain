from __future__ import annotations

from pathlib import Path

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.evidence.artifact_writer import rebuild_evidence_artifacts

from .artifact_writer import rebuild_evaluation_artifacts
from .benchmark import case_to_run_artifacts, list_benchmark_cases
from .contracts import BenchmarkCase, BenchmarkResult, RegressionSuiteResult


def run_regression_suite(
    *,
    benchmark_dir: Path | None = None,
    output_dir: Path,
    case_ids: list[str] | None = None,
) -> RegressionSuiteResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list_benchmark_cases(benchmark_dir)
    if case_ids:
        allowed = set(case_ids)
        cases = [case for case in cases if case.case_id in allowed]
    results = [run_benchmark_case(case, output_dir=output_dir) for case in cases]
    passed = len([result for result in results if result.passed])
    return RegressionSuiteResult(
        generated_at=now_iso_utc(),
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        results=results,
        artifacts_dir=str(output_dir),
    )


def run_benchmark_case(case: BenchmarkCase, *, output_dir: Path) -> BenchmarkResult:
    run_dir = output_dir / case.case_id
    case_to_run_artifacts(case, run_dir)
    try:
        rebuild_evidence_artifacts(run_dir, thread_id=case.case_id)
    except Exception:
        pass
    evaluation = rebuild_evaluation_artifacts(run_dir, thread_id=case.case_id)
    failures: list[str] = []
    criterion_by_key = {score.criterion_key: score.score for score in evaluation.criterion_scores}

    for artifact in case.expected_artifacts:
        if not (run_dir / artifact).exists():
            failures.append(f"Expected artifact missing: {artifact}")
    for entity in case.expected_entities:
        combined = f"{case.report}\n{case.notes}".lower()
        if entity.lower() not in combined:
            failures.append(f"Expected entity missing from mocked output: {entity}")
    for key, minimum in case.expected_min_scores.items():
        actual = (
            evaluation.overall_score
            if key == "overall_score"
            else criterion_by_key.get(key, 0.0)
        )
        if actual < minimum:
            failures.append(f"Score `{key}` {actual:.3f} below expected minimum {minimum:.3f}")
    warning_text = " ".join(
        [
            *evaluation.warnings,
            *(gap.description for gap in evaluation.coverage_gaps),
            *(finding.reason for finding in evaluation.hallucination_risk.findings),
        ]
    ).lower()
    for warning in case.required_warnings:
        if warning.lower() not in warning_text:
            failures.append(f"Required warning not found: {warning}")

    artifacts = sorted(
        str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
    )
    return BenchmarkResult(
        case_id=case.case_id,
        passed=not failures,
        overall_score=evaluation.overall_score,
        criterion_scores=criterion_by_key,
        warnings=evaluation.warnings,
        failures=failures,
        artifacts=artifacts,
    )
