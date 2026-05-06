from __future__ import annotations

import json
import statistics
from pathlib import Path

from .contracts import CheckSeverity, FlakyCaseSignal, QualityGateRunResult
from .governance_report import write_json


def load_recent_gate_results(
    gate_runs_dir: Path, *, gate_id: str, limit: int = 10
) -> list[QualityGateRunResult]:
    results: list[QualityGateRunResult] = []
    if not gate_runs_dir.exists():
        return results
    for path in sorted(gate_runs_dir.glob("gate-*/quality_gate_run.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("gate_id") != gate_id:
                continue
            validate = getattr(QualityGateRunResult, "model_validate", None)
            results.append(
                validate(data) if callable(validate) else QualityGateRunResult.parse_obj(data)
            )
        except Exception:
            continue
        if len(results) >= limit:
            break
    return list(reversed(results))


def detect_flaky_cases(history: list[QualityGateRunResult]) -> list[FlakyCaseSignal]:
    if len(history) < 2:
        return [
            FlakyCaseSignal(
                case_id="*",
                signal_type="unknown",
                severity=CheckSeverity.info,
                message="Not enough gate history to determine flakiness.",
                recommendation=(
                    "Run the same gate more than once before treating flakiness as known."
                ),
            )
        ]
    by_case: dict[str, list[tuple[str, float, int]]] = {}
    for result in history:
        case_scores = result.metadata.get("case_scores", {}) if result.metadata else {}
        case_statuses = result.metadata.get("case_statuses", {}) if result.metadata else {}
        case_warnings = result.metadata.get("case_warning_counts", {}) if result.metadata else {}
        for case_id, score in case_scores.items():
            by_case.setdefault(case_id, []).append(
                (
                    str(case_statuses.get(case_id, "unknown")),
                    float(score),
                    int(case_warnings.get(case_id, 0)),
                )
            )
    signals: list[FlakyCaseSignal] = []
    for case_id, values in sorted(by_case.items()):
        statuses = [status for status, _, _ in values]
        scores = [score for _, score, _ in values]
        warnings = [count for _, _, count in values]
        if {"passed", "failed"}.issubset(set(statuses)):
            signals.append(
                FlakyCaseSignal(
                    case_id=case_id,
                    signal_type="alternating_status",
                    severity=CheckSeverity.high,
                    message=f"{case_id} alternates between pass and fail.",
                    recent_statuses=statuses,
                    recent_scores=scores,
                    recommendation=(
                        "Inspect deterministic mock ordering and case artifact fingerprints."
                    ),
                )
            )
        elif len(scores) >= 2 and statistics.pstdev(scores) > 0.08:
            signals.append(
                FlakyCaseSignal(
                    case_id=case_id,
                    signal_type="score_variance",
                    severity=CheckSeverity.medium,
                    message=f"{case_id} score variance exceeds the flakiness threshold.",
                    recent_statuses=statuses,
                    recent_scores=scores,
                    recommendation="Check nondeterministic scoring inputs or artifact ordering.",
                )
            )
        elif len(warnings) >= 2 and max(warnings) - min(warnings) > 3:
            signals.append(
                FlakyCaseSignal(
                    case_id=case_id,
                    signal_type="warning_variance",
                    severity=CheckSeverity.medium,
                    message=f"{case_id} warning count varies heavily.",
                    recent_statuses=statuses,
                    recent_scores=scores,
                    recommendation="Audit warning sources and fixture lifecycle.",
                )
            )
    return signals


def write_flakiness_report(output_dir: Path, signals: list[FlakyCaseSignal]) -> list[str]:
    write_json(output_dir / "flakiness_report.json", signals)
    lines = ["# Flakiness Report", ""]
    for signal in signals:
        lines.append(f"- {signal.severity.value}: `{signal.case_id}` {signal.message}")
    (output_dir / "flakiness_report.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    return ["flakiness_report.json", "flakiness_report.md"]
