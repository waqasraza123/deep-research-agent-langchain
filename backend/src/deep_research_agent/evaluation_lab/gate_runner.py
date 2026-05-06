from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..settings import REPO_ROOT, Settings
from .baseline_store import BaselineStore, redact_mapping, warning_audit_from_baseline
from .case_loader import validate_case
from .contracts import (
    BenchmarkBaseline,
    BenchmarkRunRequest,
    BenchmarkRunResult,
    CheckSeverity,
    CheckType,
    QualityGateProfile,
    QualityGateRunRequest,
    QualityGateRunResult,
    QualityGateStatus,
    QualityGateThresholdResult,
    RegressionFindingType,
    model_to_plain,
    now_iso_utc,
)
from .coverage import analyze_coverage, write_coverage_report
from .errors import BenchmarkRunError, UnsafeBenchmarkPathError
from .flakiness import detect_flaky_cases, load_recent_gate_results, write_flakiness_report
from .gates import list_gate_profiles, resolve_gate_profile
from .governance_report import (
    render_gate_profile,
    render_governance_summary,
    render_quality_gate_run,
    render_triage,
    write_json,
)
from .regression_runner import EvaluationLabRunner
from .triage import build_triage_summary
from .warning_audit import (
    compare_warning_summaries,
    detect_warning_growth,
    summarize_warnings,
    write_warning_audit,
)


class QualityGateRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        lab_runner: EvaluationLabRunner | None = None,
        baseline_store: BaselineStore | None = None,
    ):
        self.settings = settings or Settings.load()
        self.lab_runner = lab_runner or EvaluationLabRunner(self.settings)
        self.gate_runs_dir = Path(
            getattr(self.settings, "evaluation_lab_gate_runs_dir", None)
            or (REPO_ROOT / "backend" / "benchmark_gate_runs")
        )
        self.baseline_store = baseline_store or BaselineStore(settings=self.settings)
        self.gate_runs_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[QualityGateProfile]:
        return list_gate_profiles()

    def run_gate(self, request: QualityGateRunRequest | None = None) -> QualityGateRunResult:
        request = request or QualityGateRunRequest(
            gate_id=getattr(self.settings, "evaluation_lab_default_gate_profile", "smoke")
        )
        gate_run_id = f"gate-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        gate_dir = self._safe_child(self._resolve_output_root(request.output_dir), gate_run_id)
        gate_dir.mkdir(parents=True, exist_ok=True)
        started_at = now_iso_utc()
        try:
            profile = resolve_gate_profile(request)
            write_json(gate_dir / "quality_gate_profile.json", profile)
            (gate_dir / "quality_gate_profile.md").write_text(
                render_gate_profile(profile), encoding="utf-8"
            )
            if not profile.enabled:
                result = QualityGateRunResult(
                    gate_run_id=gate_run_id,
                    gate_id=profile.gate_id,
                    status=QualityGateStatus.skipped,
                    started_at=started_at,
                    completed_at=now_iso_utc(),
                    report_artifacts=["quality_gate_profile.json", "quality_gate_profile.md"],
                    recommended_actions=["Gate is disabled."],
                )
                self._write_gate_artifacts(gate_dir, result, profile, None)
                return result

            cases = self._resolve_cases(profile, request)
            if not cases:
                result = QualityGateRunResult(
                    gate_run_id=gate_run_id,
                    gate_id=profile.gate_id,
                    status=QualityGateStatus.skipped,
                    started_at=started_at,
                    completed_at=now_iso_utc(),
                    failed_thresholds=[
                        QualityGateThresholdResult(
                            threshold_id="case_selection",
                            name="Case selection",
                            passed=False,
                            severity=CheckSeverity.high,
                            message="No benchmark cases selected.",
                        )
                    ],
                    recommended_actions=["Add case selectors or run_all for this gate."],
                )
                self._write_gate_artifacts(gate_dir, result, profile, None)
                return result

            invalid_warnings = []
            for case in cases:
                invalid_warnings.extend(validate_case(case))
            if invalid_warnings and profile.fail_on_invalid_cases:
                raise BenchmarkRunError("; ".join(invalid_warnings))

            benchmark_request = BenchmarkRunRequest(
                case_ids=[case.case_id for case in cases],
                run_all=False,
                dry_run=request.dry_run,
                use_mock_agent=profile.allow_mock_agent,
                use_offline_fetcher=profile.use_offline_fetcher,
                settings_overrides=request.settings_overrides,
            )
            benchmark = self.lab_runner.run_cases(benchmark_request)
            baseline = self._load_baseline(profile, request)
            regressions = []
            improvements = []
            comparison_artifacts: list[str] = []
            if baseline is not None:
                regressions, improvements = self.baseline_store.compare_result_to_baseline(
                    benchmark, baseline
                )
                comparison_artifacts = self.baseline_store.write_comparison_artifacts(
                    gate_dir,
                    baseline=baseline,
                    regressions=regressions,
                    improvements=improvements,
                )

            warning_texts = [
                warning for case in benchmark.case_results for warning in case.warnings
            ]
            current_warnings = summarize_warnings(
                warning_texts,
                max_warning_count=profile.max_warning_count,
                max_serious_warnings=profile.max_serious_warnings,
            )
            baseline_warnings = warning_audit_from_baseline(baseline)
            warning_audit = compare_warning_summaries(current_warnings, baseline_warnings)
            if detect_warning_growth(
                warning_audit, baseline_warnings, max_growth_ratio=profile.max_warning_growth_ratio
            ):
                warning_audit.budget_exceeded = True
            coverage = analyze_coverage(cases)
            thresholds = self._evaluate_thresholds(benchmark, profile, regressions, warning_audit)
            failed_thresholds = [threshold for threshold in thresholds if not threshold.passed]
            passed_thresholds = [threshold for threshold in thresholds if threshold.passed]
            history = load_recent_gate_results(self.gate_runs_dir, gate_id=profile.gate_id)
            flaky = detect_flaky_cases(
                [*history, self._history_stub(gate_run_id, profile, benchmark)]
            )
            if flaky and flaky[0].signal_type == "unknown":
                flaky = []
            if profile.fail_on_flaky_cases and flaky:
                failed_thresholds.append(
                    QualityGateThresholdResult(
                        threshold_id="flaky_cases",
                        name="No flaky cases",
                        passed=False,
                        expected=0,
                        actual=len(flaky),
                        severity=CheckSeverity.high,
                        message=f"{len(flaky)} flaky case signals detected.",
                        affected_cases=[signal.case_id for signal in flaky],
                        recommendation="Stabilize deterministic benchmark outputs before merging.",
                    )
                )
            status = QualityGateStatus.passed
            if any(
                threshold.severity in {CheckSeverity.high, CheckSeverity.critical}
                for threshold in failed_thresholds
            ):
                status = QualityGateStatus.failed
            elif failed_thresholds:
                status = QualityGateStatus.warning
            triage = build_triage_summary(benchmark, regressions, warning_audit)
            recommended = self._recommended_actions(failed_thresholds, regressions, warning_audit)
            metadata = {
                "case_scores": {case.case_id: case.score for case in benchmark.case_results},
                "case_statuses": {case.case_id: case.status for case in benchmark.case_results},
                "case_warning_counts": {
                    case.case_id: len(case.warnings) for case in benchmark.case_results
                },
                "settings_overrides_redacted": redact_mapping(request.settings_overrides),
            }
            result = QualityGateRunResult(
                gate_run_id=gate_run_id,
                gate_id=profile.gate_id,
                status=status,
                started_at=started_at,
                completed_at=now_iso_utc(),
                benchmark_run_id=benchmark.run_id,
                baseline_id=baseline.baseline_id if baseline else profile.baseline_id,
                total_cases=benchmark.total_cases,
                passed_cases=benchmark.passed_cases,
                failed_cases=benchmark.failed_cases,
                errored_cases=benchmark.errored_cases,
                skipped_cases=benchmark.skipped_cases,
                pass_rate=benchmark.passed_cases / benchmark.total_cases
                if benchmark.total_cases
                else 0.0,
                average_score=benchmark.average_score,
                minimum_case_score=min(
                    (case.score for case in benchmark.case_results), default=0.0
                ),
                failed_thresholds=failed_thresholds,
                passed_thresholds=passed_thresholds,
                regressions=regressions,
                improvements=improvements,
                flaky_cases=flaky,
                warning_audit=warning_audit,
                coverage_summary=coverage,
                triage_summary=triage,
                report_artifacts=[
                    "quality_gate_profile.json",
                    "quality_gate_profile.md",
                    *comparison_artifacts,
                    "warning_audit.json",
                    "warning_audit.md",
                    "benchmark_coverage.json",
                    "benchmark_coverage.md",
                    "flakiness_report.json",
                    "flakiness_report.md",
                    "quality_gate_run.json",
                    "quality_gate_run.md",
                    "quality_gate_thresholds.json",
                    "quality_gate_thresholds.md",
                    "quality_gate_triage.json",
                    "quality_gate_triage.md",
                    "quality_gate_recommendations.md",
                    "governance_summary.json",
                    "governance_summary.md",
                ],
                recommended_actions=recommended,
                metadata=metadata,
            )
            write_warning_audit(gate_dir, warning_audit)
            write_coverage_report(gate_dir, coverage)
            write_flakiness_report(gate_dir, flaky)
            self._write_gate_artifacts(gate_dir, result, profile, baseline)
            if request.update_baseline and result.status == QualityGateStatus.passed:
                self.baseline_store.promote_baseline(
                    benchmark,
                    gate_id=profile.gate_id,
                    name=f"{profile.name} baseline",
                    description=f"Promoted from gate run {gate_run_id}",
                    metadata={"gate_run_id": gate_run_id, **request.metadata},
                )
            return result
        except Exception as exc:
            result = QualityGateRunResult(
                gate_run_id=gate_run_id,
                gate_id=request.gate_id,
                status=QualityGateStatus.errored,
                started_at=started_at,
                completed_at=now_iso_utc(),
                failed_thresholds=[
                    QualityGateThresholdResult(
                        threshold_id="gate_error",
                        name="Gate execution error",
                        passed=False,
                        expected="successful gate execution",
                        actual=f"{type(exc).__name__}: {exc}",
                        severity=CheckSeverity.critical,
                        message=f"Gate execution errored: {type(exc).__name__}: {exc}",
                        recommendation="Fix gate configuration or runner error.",
                    )
                ],
                recommended_actions=[f"Fix gate error: {type(exc).__name__}: {exc}"],
                metadata={
                    "settings_overrides_redacted": redact_mapping(request.settings_overrides)
                },
            )
            self._write_gate_artifacts(gate_dir, result, None, None)
            return result

    def read_gate_run(self, gate_run_id: str) -> QualityGateRunResult:
        self._validate_run_id(gate_run_id)
        path = self._safe_child(self.gate_runs_dir, Path(gate_run_id) / "quality_gate_run.json")
        if not path.exists():
            raise BenchmarkRunError(f"Unknown quality gate run: {gate_run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        validate = getattr(QualityGateRunResult, "model_validate", None)
        return validate(data) if callable(validate) else QualityGateRunResult.parse_obj(data)

    def read_governance_summary(self, gate_run_id: str, *, markdown: bool = False):
        self._validate_run_id(gate_run_id)
        filename = "governance_summary.md" if markdown else "governance_summary.json"
        path = self._safe_child(self.gate_runs_dir, Path(gate_run_id) / filename)
        if not path.exists():
            raise BenchmarkRunError(f"Unknown quality gate summary: {gate_run_id}")
        if markdown:
            return path.read_text(encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_triage(self, gate_run_id: str):
        self._validate_run_id(gate_run_id)
        path = self._safe_child(self.gate_runs_dir, Path(gate_run_id) / "quality_gate_triage.json")
        if not path.exists():
            raise BenchmarkRunError(f"Unknown quality gate triage: {gate_run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_cases(self, profile: QualityGateProfile, request: QualityGateRunRequest):
        categories = [category.value for category in profile.categories]
        cases = self.lab_runner.list_cases(
            case_ids=[] if request.run_all else profile.case_ids,
            categories=[] if request.run_all else categories,
            tags=[] if request.run_all else profile.tags,
        )
        if request.run_all or not (profile.case_ids or profile.categories or profile.tags):
            cases = self.lab_runner.list_cases()
        excluded = set(profile.excluded_case_ids)
        return [case for case in cases if case.case_id not in excluded]

    def _load_baseline(
        self, profile: QualityGateProfile, request: QualityGateRunRequest
    ) -> BenchmarkBaseline | None:
        compare = (
            request.compare_against_baseline
            if request.compare_against_baseline is not None
            else profile.compare_against_baseline
        )
        if not compare:
            return None
        baseline_id = request.baseline_id or profile.baseline_id
        baseline = (
            self.baseline_store.get_baseline(baseline_id)
            if baseline_id
            else self.baseline_store.get_latest_baseline(profile.gate_id)
        )
        if baseline is None and profile.fail_on_missing_baseline:
            raise BenchmarkRunError(f"Missing required baseline for gate {profile.gate_id}")
        return baseline

    def _evaluate_thresholds(
        self,
        benchmark: BenchmarkRunResult,
        profile: QualityGateProfile,
        regressions,
        warning_audit,
    ) -> list[QualityGateThresholdResult]:
        total = benchmark.total_cases or 1
        pass_rate = benchmark.passed_cases / total
        min_score = min((case.score for case in benchmark.case_results), default=0.0)
        critical_failures = [
            (case.case_id, check)
            for case in benchmark.case_results
            for check in case.check_results
            if not check.passed and check.severity == CheckSeverity.critical
        ]
        high_failures = [
            (case.case_id, check)
            for case in benchmark.case_results
            for check in case.check_results
            if not check.passed and check.severity == CheckSeverity.high
        ]
        missed_critical = [trap for case in benchmark.case_results for trap in case.missed_traps]
        thresholds: list[QualityGateThresholdResult] = []
        self._add_threshold(
            thresholds,
            "minimum_average_score",
            benchmark.average_score >= profile.minimum_average_score,
            profile.minimum_average_score,
            round(benchmark.average_score, 4),
            "Average score",
            f"Average score must be at least {profile.minimum_average_score:.3f}.",
        )
        self._add_threshold(
            thresholds,
            "minimum_pass_rate",
            pass_rate >= profile.minimum_pass_rate,
            profile.minimum_pass_rate,
            round(pass_rate, 4),
            "Pass rate",
            f"Pass rate must be at least {profile.minimum_pass_rate:.3f}.",
        )
        if profile.minimum_case_score is not None:
            self._add_threshold(
                thresholds,
                "minimum_case_score",
                min_score >= profile.minimum_case_score,
                profile.minimum_case_score,
                round(min_score, 4),
                "Minimum case score",
                f"No case may score below {profile.minimum_case_score:.3f}.",
                [
                    case.case_id
                    for case in benchmark.case_results
                    if case.score < profile.minimum_case_score
                ],
            )
        if profile.max_failed_cases is not None:
            self._add_threshold(
                thresholds,
                "max_failed_cases",
                benchmark.failed_cases <= profile.max_failed_cases,
                profile.max_failed_cases,
                benchmark.failed_cases,
                "Failed cases",
                f"Failed cases must be <= {profile.max_failed_cases}.",
                [case.case_id for case in benchmark.case_results if not case.passed],
            )
        if profile.max_critical_failures is not None:
            self._add_threshold(
                thresholds,
                "max_critical_failures",
                len(critical_failures) <= profile.max_critical_failures,
                profile.max_critical_failures,
                len(critical_failures),
                "Critical failures",
                f"Critical failures must be <= {profile.max_critical_failures}.",
                [case_id for case_id, _ in critical_failures],
                CheckSeverity.critical,
            )
        if profile.max_high_severity_failures is not None:
            self._add_threshold(
                thresholds,
                "max_high_severity_failures",
                len(high_failures) <= profile.max_high_severity_failures,
                profile.max_high_severity_failures,
                len(high_failures),
                "High severity failures",
                f"High severity failures must be <= {profile.max_high_severity_failures}.",
                [case_id for case_id, _ in high_failures],
            )
        if profile.max_missed_critical_traps is not None:
            self._add_threshold(
                thresholds,
                "max_missed_critical_traps",
                len(missed_critical) <= profile.max_missed_critical_traps,
                profile.max_missed_critical_traps,
                len(missed_critical),
                "Missed critical traps",
                f"Missed critical traps must be <= {profile.max_missed_critical_traps}.",
                missed_critical,
                CheckSeverity.critical,
            )
        if profile.max_new_regressions is not None and regressions:
            self._add_threshold(
                thresholds,
                "max_new_regressions",
                len(regressions) <= profile.max_new_regressions,
                profile.max_new_regressions,
                len(regressions),
                "New regressions",
                f"New regressions must be <= {profile.max_new_regressions}.",
                [finding.case_id for finding in regressions],
            )
        if profile.require_no_prompt_injection_failures:
            failures = [
                case.case_id
                for case in benchmark.case_results
                for check in case.check_results
                if not check.passed and check.check_type == CheckType.prompt_injection_resistance
            ]
            self._add_threshold(
                thresholds,
                "no_prompt_injection_failures",
                not failures,
                0,
                len(failures),
                "Prompt injection failures",
                "No prompt-injection failures are allowed.",
                failures,
                CheckSeverity.critical,
            )
        if profile.require_no_artifact_integrity_failures:
            failures = [
                case.case_id
                for case in benchmark.case_results
                for check in case.check_results
                if not check.passed
                and check.check_type
                in {
                    CheckType.artifact_exists,
                    CheckType.artifact_nonempty,
                    CheckType.artifact_valid_json,
                }
            ]
            self._add_threshold(
                thresholds,
                "no_artifact_integrity_failures",
                not failures,
                0,
                len(failures),
                "Artifact integrity failures",
                "No required artifact integrity failures are allowed.",
                failures,
            )
        regression_type_checks = [
            (
                profile.require_no_numeric_regressions,
                RegressionFindingType.new_numeric_failure,
                "numeric",
            ),
            (
                profile.require_no_temporal_regressions,
                RegressionFindingType.new_temporal_failure,
                "temporal",
            ),
            (
                profile.require_no_citation_regressions,
                RegressionFindingType.new_citation_failure,
                "citation",
            ),
        ]
        for required, finding_type, label in regression_type_checks:
            if not required:
                continue
            findings = [finding for finding in regressions if finding.type == finding_type]
            self._add_threshold(
                thresholds,
                f"no_{label}_regressions",
                not findings,
                0,
                len(findings),
                f"{label.title()} regressions",
                f"No {label} regressions are allowed.",
                [finding.case_id for finding in findings],
            )
        if profile.max_warning_count is not None:
            self._add_threshold(
                thresholds,
                "max_warning_count",
                warning_audit.total_warnings <= profile.max_warning_count,
                profile.max_warning_count,
                warning_audit.total_warnings,
                "Warning count",
                f"Warnings must be <= {profile.max_warning_count}.",
                severity=CheckSeverity.medium,
            )
        if profile.max_serious_warnings is not None:
            self._add_threshold(
                thresholds,
                "max_serious_warnings",
                warning_audit.serious_warning_count <= profile.max_serious_warnings,
                profile.max_serious_warnings,
                warning_audit.serious_warning_count,
                "Serious warnings",
                f"Serious warnings must be <= {profile.max_serious_warnings}.",
                severity=CheckSeverity.high,
            )
        if warning_audit.budget_exceeded:
            self._add_threshold(
                thresholds,
                "warning_budget",
                False,
                "within warning budget",
                "exceeded",
                "Warning budget",
                "Warning budget was exceeded.",
                severity=CheckSeverity.high,
            )
        return thresholds

    def _add_threshold(
        self,
        thresholds: list[QualityGateThresholdResult],
        threshold_id: str,
        passed: bool,
        expected,
        actual,
        name: str,
        message: str,
        affected_cases: list[str] | None = None,
        severity: CheckSeverity = CheckSeverity.high,
    ) -> None:
        thresholds.append(
            QualityGateThresholdResult(
                threshold_id=threshold_id,
                name=name,
                passed=passed,
                expected=expected,
                actual=actual,
                severity=severity,
                message=message if passed else f"{message} Actual: {actual}.",
                affected_cases=sorted(set(affected_cases or [])),
                recommendation="Inspect failed case artifacts and compare against the baseline.",
            )
        )

    def _write_gate_artifacts(
        self,
        gate_dir: Path,
        result: QualityGateRunResult,
        profile: QualityGateProfile | None,
        baseline: BenchmarkBaseline | None,
    ) -> None:
        write_json(gate_dir / "quality_gate_run.json", result)
        (gate_dir / "quality_gate_run.md").write_text(
            render_quality_gate_run(result, profile), encoding="utf-8"
        )
        write_json(
            gate_dir / "quality_gate_thresholds.json",
            result.failed_thresholds + result.passed_thresholds,
        )
        lines = ["# Quality Gate Thresholds", ""]
        for threshold in [*result.failed_thresholds, *result.passed_thresholds]:
            state = "PASS" if threshold.passed else "FAIL"
            lines.append(f"- {state} `{threshold.threshold_id}`: {threshold.message}")
        (gate_dir / "quality_gate_thresholds.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )
        write_json(gate_dir / "quality_gate_triage.json", result.triage_summary)
        (gate_dir / "quality_gate_triage.md").write_text(
            render_triage(result.triage_summary), encoding="utf-8"
        )
        (gate_dir / "quality_gate_recommendations.md").write_text(
            "# Quality Gate Recommendations\n\n"
            + "\n".join(
                f"- {item}" for item in result.recommended_actions or ["No action required"]
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(gate_dir / "governance_summary.json", {"result": model_to_plain(result)})
        (gate_dir / "governance_summary.md").write_text(
            render_governance_summary(result, baseline), encoding="utf-8"
        )

    def _recommended_actions(self, failed_thresholds, regressions, warning_audit) -> list[str]:
        actions = [
            threshold.recommendation for threshold in failed_thresholds if threshold.recommendation
        ]
        if regressions:
            actions.append(
                "Compare current artifacts with the promoted baseline before changing thresholds."
            )
        if warning_audit.budget_exceeded:
            actions.extend(warning_audit.recommendations)
        return sorted(set(actions))[:12] or ["No action required."]

    def _history_stub(
        self, gate_run_id, profile, benchmark: BenchmarkRunResult
    ) -> QualityGateRunResult:
        return QualityGateRunResult(
            gate_run_id=gate_run_id,
            gate_id=profile.gate_id,
            status=QualityGateStatus.passed
            if benchmark.status == "passed"
            else QualityGateStatus.failed,
            started_at=benchmark.started_at,
            completed_at=benchmark.completed_at,
            benchmark_run_id=benchmark.run_id,
            metadata={
                "case_scores": {case.case_id: case.score for case in benchmark.case_results},
                "case_statuses": {case.case_id: case.status for case in benchmark.case_results},
                "case_warning_counts": {
                    case.case_id: len(case.warnings) for case in benchmark.case_results
                },
            },
        )

    def _resolve_output_root(self, output_dir: str | None) -> Path:
        root = self.gate_runs_dir.resolve()
        if not output_dir:
            root.mkdir(parents=True, exist_ok=True)
            return root
        candidate = Path(output_dir)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate != root and root not in candidate.parents:
            raise UnsafeBenchmarkPathError(
                "Gate output_dir must stay under evaluation_lab_gate_runs_dir"
            )
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _safe_child(self, root: Path, rel: str | Path) -> Path:
        resolved_root = root.resolve()
        path = (resolved_root / rel).resolve()
        if path != resolved_root and resolved_root not in path.parents:
            raise UnsafeBenchmarkPathError(f"Path escapes quality gate root: {rel}")
        return path

    def _validate_run_id(self, value: str) -> None:
        if not value.startswith("gate-") or "/" in value or "\\" in value or ".." in value:
            raise UnsafeBenchmarkPathError(f"Unsafe gate run ID: {value}")
