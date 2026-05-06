from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from ..settings import REPO_ROOT, Settings
from .contracts import (
    BaselineCaseSnapshot,
    BenchmarkBaseline,
    BenchmarkCaseResult,
    BenchmarkRunResult,
    CheckSeverity,
    CheckType,
    ImprovementFinding,
    RegressionFinding,
    RegressionFindingType,
    WarningAudit,
    model_to_plain,
    now_iso_utc,
)
from .errors import BenchmarkRunError, UnsafeBenchmarkPathError
from .governance_report import write_json

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in values.items():
        lowered = str(key).lower()
        if any(secret in lowered for secret in ("key", "secret", "token", "password")):
            redacted[str(key)] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[str(key)] = redact_mapping(value)
        else:
            redacted[str(key)] = value
    return redacted


class BaselineStore:
    def __init__(self, root_dir: Path | None = None, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        configured = root_dir or getattr(self.settings, "evaluation_lab_baselines_dir", None)
        self.root_dir = Path(configured or (REPO_ROOT / "backend" / "benchmarks" / "baselines"))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "history").mkdir(parents=True, exist_ok=True)

    def create_baseline_from_run(
        self,
        result: BenchmarkRunResult,
        *,
        name: str,
        description: str = "",
        gate_id: str | None = None,
        baseline_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkBaseline:
        baseline_id = baseline_id or f"{gate_id or 'baseline'}-{result.run_id}"
        self._validate_id(baseline_id)
        snapshots = [self._case_snapshot(case) for case in result.case_results]
        pass_rate = result.passed_cases / result.total_cases if result.total_cases else 0.0
        baseline = BenchmarkBaseline(
            baseline_id=baseline_id,
            name=name,
            description=description,
            created_at=now_iso_utc(),
            created_from_run_id=result.run_id,
            gate_id=gate_id,
            git_commit=self._git_commit(),
            case_results=snapshots,
            suite_score=result.average_score,
            pass_rate=pass_rate,
            average_score=result.average_score,
            case_scores={case.case_id: case.score for case in result.case_results},
            check_fingerprints={
                case.case_id: self._fingerprint(case.check_results) for case in result.case_results
            },
            warning_fingerprint=self._fingerprint(
                [warning for case in result.case_results for warning in case.warnings]
            ),
            metadata=redact_mapping(metadata or {}),
        )
        self.write_baseline(baseline)
        if gate_id:
            self._write_alias(gate_id, baseline)
        self._write_alias("latest", baseline)
        return baseline

    def write_baseline(self, baseline: BenchmarkBaseline) -> None:
        self._validate_id(baseline.baseline_id)
        path = self._baseline_path(baseline.baseline_id)
        write_json(path, baseline)
        write_json(
            self._safe_child(self.root_dir / "history", f"{baseline.baseline_id}.json"), baseline
        )
        self._ensure_readme()

    def get_baseline(self, baseline_id: str) -> BenchmarkBaseline:
        self._validate_id(baseline_id)
        path = self._baseline_path(baseline_id)
        if not path.exists():
            raise BenchmarkRunError(f"Unknown benchmark baseline: {baseline_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        validate = getattr(BenchmarkBaseline, "model_validate", None)
        return validate(data) if callable(validate) else BenchmarkBaseline.parse_obj(data)

    def get_latest_baseline(self, gate_id: str | None = None) -> BenchmarkBaseline | None:
        for baseline_id in [gate_id, "latest"]:
            if not baseline_id:
                continue
            try:
                return self.get_baseline(baseline_id)
            except BenchmarkRunError:
                continue
        return None

    def list_baselines(self) -> list[BenchmarkBaseline]:
        baselines = []
        for path in sorted(self.root_dir.glob("*.json")):
            try:
                baselines.append(self.get_baseline(path.stem))
            except Exception:
                continue
        return baselines

    def promote_baseline(
        self,
        result: BenchmarkRunResult,
        *,
        gate_id: str,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkBaseline:
        return self.create_baseline_from_run(
            result,
            name=name,
            description=description,
            gate_id=gate_id,
            baseline_id=gate_id,
            metadata=metadata,
        )

    def compare_result_to_baseline(
        self, current: BenchmarkRunResult, baseline: BenchmarkBaseline
    ) -> tuple[list[RegressionFinding], list[ImprovementFinding]]:
        by_baseline = {case.case_id: case for case in baseline.case_results}
        regressions: list[RegressionFinding] = []
        improvements: list[ImprovementFinding] = []
        for current_case in current.case_results:
            previous = by_baseline.get(current_case.case_id)
            if previous is None:
                continue
            delta = current_case.score - previous.score
            if previous.passed and not current_case.passed:
                regressions.append(
                    self._regression(
                        current_case,
                        RegressionFindingType.newly_failed_case,
                        "high",
                        f"{current_case.case_id} newly failed.",
                        previous.passed,
                        current_case.passed,
                        delta,
                    )
                )
            if delta <= -0.05:
                regressions.append(
                    self._regression(
                        current_case,
                        RegressionFindingType.score_drop,
                        "medium",
                        f"{current_case.case_id} score dropped by {abs(delta):.3f}.",
                        previous.score,
                        current_case.score,
                        delta,
                    )
                )
            new_missed = sorted(set(current_case.missed_traps) - set(previous.missed_traps))
            for trap_id in new_missed:
                regressions.append(
                    self._regression(
                        current_case,
                        RegressionFindingType.newly_missed_trap,
                        "critical",
                        f"{current_case.case_id} newly missed trap `{trap_id}`.",
                        previous.missed_traps,
                        current_case.missed_traps,
                        delta,
                    )
                )
            for check in current_case.check_results:
                if check.passed:
                    continue
                previous_failed = previous.check_results_summary.get(
                    f"{check.check_type.value}:failed", 0
                )
                if previous_failed:
                    continue
                finding_type = self._finding_type_for_check(check.check_type)
                if finding_type is not None:
                    regressions.append(
                        self._regression(
                            current_case,
                            finding_type,
                            check.severity.value,
                            f"{current_case.case_id}: {check.name} failed.",
                            "passed",
                            check.message,
                            delta,
                            [check.check_id],
                        )
                    )
            if not previous.passed and current_case.passed:
                improvements.append(
                    ImprovementFinding(
                        finding_id=f"{current_case.case_id}-newly-passed",
                        case_id=current_case.case_id,
                        type="newly_passed_case",
                        message=f"{current_case.case_id} now passes.",
                        baseline_value=previous.passed,
                        current_value=current_case.passed,
                        score_delta=delta,
                    )
                )
            elif delta >= 0.05:
                improvements.append(
                    ImprovementFinding(
                        finding_id=f"{current_case.case_id}-score-improved",
                        case_id=current_case.case_id,
                        type="score_improvement",
                        message=f"{current_case.case_id} score improved by {delta:.3f}.",
                        baseline_value=previous.score,
                        current_value=current_case.score,
                        score_delta=delta,
                    )
                )
        return regressions, improvements

    def write_comparison_artifacts(
        self,
        output_dir: Path,
        *,
        baseline: BenchmarkBaseline,
        regressions: list[RegressionFinding],
        improvements: list[ImprovementFinding],
    ) -> list[str]:
        payload = {
            "baseline_id": baseline.baseline_id,
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "regressions": model_to_plain(regressions),
            "improvements": model_to_plain(improvements),
        }
        write_json(output_dir / "baseline_comparison.json", payload)
        write_json(output_dir / "regression_findings.json", regressions)
        write_json(output_dir / "improvement_findings.json", improvements)
        self._write_findings_md(
            output_dir / "baseline_comparison.md", "Baseline Comparison", payload
        )
        self._write_findings_md(
            output_dir / "regression_findings.md", "Regression Findings", regressions
        )
        self._write_findings_md(
            output_dir / "improvement_findings.md", "Improvement Findings", improvements
        )
        return [
            "baseline_comparison.json",
            "baseline_comparison.md",
            "regression_findings.json",
            "regression_findings.md",
            "improvement_findings.json",
            "improvement_findings.md",
        ]

    def _case_snapshot(self, case: BenchmarkCaseResult) -> BaselineCaseSnapshot:
        return BaselineCaseSnapshot(
            case_id=case.case_id,
            status=case.status,
            passed=case.passed,
            score=case.score,
            check_results_summary=self._check_summary(case),
            missed_traps=case.missed_traps,
            detected_traps=case.detected_traps,
            warning_count=len(case.warnings),
            artifact_hashes={
                artifact: self._fingerprint(artifact) for artifact in case.artifact_paths
            },
            report_fingerprint=self._fingerprint(case.failure_reasons + case.missed_traps),
        )

    def _check_summary(self, case: BenchmarkCaseResult) -> dict[str, int]:
        summary: dict[str, int] = {}
        for check in case.check_results:
            key = f"{check.check_type.value}:{'passed' if check.passed else 'failed'}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def _regression(
        self,
        case: BenchmarkCaseResult,
        finding_type: RegressionFindingType,
        severity: str,
        message: str,
        baseline_value: Any,
        current_value: Any,
        score_delta: float,
        affected_checks: list[str] | None = None,
    ) -> RegressionFinding:
        return RegressionFinding(
            finding_id=f"{case.case_id}-{finding_type.value}-{abs(hash(message)) % 100000}",
            case_id=case.case_id,
            category=finding_type.value,
            severity=CheckSeverity(severity)
            if severity in CheckSeverity._value2member_map_
            else CheckSeverity.high,
            type=finding_type,
            message=message,
            baseline_value=baseline_value,
            current_value=current_value,
            score_delta=score_delta,
            affected_checks=affected_checks or [],
            recommendation="Inspect benchmark artifacts and restore source-grounded behavior.",
        )

    def _finding_type_for_check(self, check_type: CheckType) -> RegressionFindingType | None:
        if check_type in {
            CheckType.artifact_exists,
            CheckType.artifact_nonempty,
            CheckType.artifact_valid_json,
        }:
            return RegressionFindingType.new_artifact_failure
        if check_type == CheckType.prompt_injection_resistance:
            return RegressionFindingType.new_prompt_injection_failure
        if check_type == CheckType.numeric_support:
            return RegressionFindingType.new_numeric_failure
        if check_type in {CheckType.date_support, CheckType.stale_source_warning}:
            return RegressionFindingType.new_temporal_failure
        if check_type in {CheckType.citation_support, CheckType.source_traceability}:
            return RegressionFindingType.new_citation_failure
        return None

    def _baseline_path(self, baseline_id: str) -> Path:
        return self._safe_child(self.root_dir, f"{baseline_id}.json")

    def _safe_child(self, root: Path, rel: str | Path) -> Path:
        resolved_root = root.resolve()
        path = (resolved_root / rel).resolve()
        if path != resolved_root and resolved_root not in path.parents:
            raise UnsafeBenchmarkPathError(f"Path escapes baseline root: {rel}")
        return path

    def _validate_id(self, value: str) -> None:
        if not value or not SAFE_ID.fullmatch(value):
            raise UnsafeBenchmarkPathError(f"Unsafe baseline ID: {value}")

    def _write_alias(self, alias: str, baseline: BenchmarkBaseline) -> None:
        self._validate_id(alias)
        data = baseline.to_json_dict()
        data["baseline_id"] = alias
        write_json(self._baseline_path(alias), data)

    def _ensure_readme(self) -> None:
        readme = self.root_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Evaluation Lab Baselines\n\n"
                "Baselines are compact JSON snapshots promoted from benchmark or "
                "quality gate runs. "
                "Promote them intentionally after reviewing governance reports.\n",
                encoding="utf-8",
            )

    def _write_findings_md(self, path: Path, title: str, payload: Any) -> None:
        lines = [f"# {title}", ""]
        if isinstance(payload, list):
            lines.extend(f"- {item.message}" for item in payload)
            if not payload:
                lines.append("- None")
        else:
            lines.extend(
                [
                    f"- Baseline: `{payload['baseline_id']}`",
                    f"- Regressions: {payload['regression_count']}",
                    f"- Improvements: {payload['improvement_count']}",
                ]
            )
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _git_commit(self) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    def _fingerprint(self, value: Any) -> str:
        encoded = json.dumps(model_to_plain(value), sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def warning_audit_from_baseline(baseline: BenchmarkBaseline | None) -> WarningAudit | None:
    if baseline is None:
        return None
    count = sum(case.warning_count for case in baseline.case_results)
    return WarningAudit(total_warnings=count, warning_fingerprint=baseline.warning_fingerprint)  # type: ignore[call-arg]
