from __future__ import annotations

import json
from pathlib import Path

from .contracts import BenchmarkCase, CheckResult, CheckSeverity, CheckType
from .expected_outputs import _write_checks, normalize_text


def _check(
    check_id: str,
    check_type: CheckType,
    name: str,
    passed: bool,
    *,
    severity: CheckSeverity = CheckSeverity.medium,
    message: str = "",
    expected=None,
    actual=None,
    artifact: str = "",
    recommendation: str = "",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        check_type=check_type,
        name=name,
        passed=passed,
        score=1.0 if passed else 0.0,
        severity=severity,
        message=message or ("passed" if passed else "failed"),
        expected=expected,
        actual=actual,
        affected_artifacts=[artifact] if artifact else [],
        recommendation=recommendation,
    )


def check_required_artifacts(run_dir: Path, case: BenchmarkCase) -> list[CheckResult]:
    checks = []
    for artifact in case.expected.required_artifacts:
        path = run_dir / artifact
        checks.append(
            _check(
                f"artifact_exists_{artifact.replace('/', '_')}",
                CheckType.artifact_exists,
                f"Required artifact `{artifact}` exists",
                path.exists() and path.is_file(),
                severity=CheckSeverity.critical,
                expected=artifact,
                actual="exists" if path.exists() else "missing",
                artifact=artifact,
                recommendation=(
                    "Ensure the agent or backfill step writes every guaranteed artifact."
                ),
            )
        )
    return checks


def check_json_artifacts(run_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for path in sorted(run_dir.rglob("*.json")):
        rel = str(path.relative_to(run_dir)).replace("\\", "/")
        try:
            json.loads(path.read_text(encoding="utf-8"))
            passed = True
            message = "valid JSON"
        except Exception as exc:
            passed = False
            message = f"invalid JSON: {type(exc).__name__}: {exc}"
        checks.append(
            _check(
                f"json_valid_{rel.replace('/', '_')}",
                CheckType.artifact_valid_json,
                f"JSON artifact `{rel}` parses",
                passed,
                severity=CheckSeverity.high,
                message=message,
                artifact=rel,
            )
        )
    for path in sorted(run_dir.rglob("*.jsonl")):
        rel = str(path.relative_to(run_dir)).replace("\\", "/")
        passed = True
        message = "valid JSONL"
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as exc:
                passed = False
                message = f"invalid JSONL line {line_no}: {type(exc).__name__}: {exc}"
                break
        checks.append(
            _check(
                f"jsonl_valid_{rel.replace('/', '_')}",
                CheckType.artifact_valid_json,
                f"JSONL artifact `{rel}` parses",
                passed,
                severity=CheckSeverity.high,
                message=message,
                artifact=rel,
            )
        )
    return checks


def check_markdown_artifacts(run_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for path in sorted(run_dir.rglob("*.md")):
        rel = str(path.relative_to(run_dir)).replace("\\", "/")
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        meaningful = len(text) >= 20 and "agent did not write" not in normalize_text(text)
        checks.append(
            _check(
                f"markdown_nonempty_{rel.replace('/', '_')}",
                CheckType.artifact_nonempty,
                f"Markdown artifact `{rel}` is meaningful",
                meaningful,
                severity=CheckSeverity.medium
                if rel in {"plan.md", "notes.md", "report.md"}
                else CheckSeverity.low,
                expected="non-placeholder markdown",
                actual=f"{len(text)} chars",
                artifact=rel,
                recommendation="Write useful markdown content rather than placeholder text.",
            )
        )
    return checks


def check_source_traceability(run_dir: Path, case: BenchmarkCase) -> list[CheckResult]:
    path = run_dir / "sources.json"
    checks: list[CheckResult] = []
    source_text = ""
    sources = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                sources = [item for item in loaded if isinstance(item, dict)]
            source_text = json.dumps(loaded)
        except Exception:
            source_text = path.read_text(encoding="utf-8", errors="ignore")
    for source in case.local_sources:
        present = source.source_id in source_text or source.url in source_text
        checks.append(
            _check(
                f"source_trace_{source.source_id}",
                CheckType.source_traceability,
                f"Source `{source.source_id}` is traceable",
                present,
                severity=CheckSeverity.high,
                expected=source.url,
                actual="present" if present else "missing",
                artifact="sources.json",
            )
        )
    report = (
        (run_dir / "report.md").read_text(encoding="utf-8", errors="ignore")
        if (run_dir / "report.md").exists()
        else ""
    )
    unknown_citations = []
    for token in set(part.split("]", 1)[0] for part in report.split("[")[1:] if "]" in part):
        if token.startswith("S") and not any(
            token == str(item.get("source_id")) for item in sources
        ):
            unknown_citations.append(token)
    checks.append(
        _check(
            "unknown_report_sources",
            CheckType.source_traceability,
            "Report cites only known sources",
            not unknown_citations,
            severity=CheckSeverity.high,
            expected="known source ids",
            actual=unknown_citations,
            artifact="report.md",
        )
    )
    return checks


def check_artifact_safety(run_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    root = run_dir.resolve()
    unsafe = []
    for path in run_dir.rglob("*"):
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            unsafe.append(str(path))
    checks.append(
        _check(
            "artifact_path_safety",
            CheckType.artifact_exists,
            "Artifact paths stay inside run directory",
            not unsafe,
            severity=CheckSeverity.critical,
            expected="no escaping paths",
            actual=unsafe,
            recommendation=(
                "Reject traversal and symlink escapes before writing or reading artifacts."
            ),
        )
    )
    return checks


def check_mock_marker(run_dir: Path, *, use_mock_agent: bool) -> list[CheckResult]:
    if not use_mock_agent:
        return []
    combined = ""
    for rel in ("plan.md", "notes.md", "report.md", "metadata.json"):
        path = run_dir / rel
        if path.exists():
            combined += "\n" + path.read_text(encoding="utf-8", errors="ignore")
    present = "mock" in normalize_text(combined)
    return [
        _check(
            "mock_mode_marker",
            CheckType.report_completeness,
            "Mock mode is clearly marked",
            present,
            severity=CheckSeverity.high,
            expected="mock marker",
            actual="present" if present else "missing",
            recommendation=(
                "Mark deterministic mock output so it is not confused with live research."
            ),
        )
    ]


def run_artifact_checks(
    case: BenchmarkCase,
    run_dir: Path,
    *,
    use_mock_agent: bool = True,
) -> list[CheckResult]:
    checks = [
        *check_required_artifacts(run_dir, case),
        *check_json_artifacts(run_dir),
        *check_markdown_artifacts(run_dir),
        *check_source_traceability(run_dir, case),
        *check_artifact_safety(run_dir),
        *check_mock_marker(run_dir, use_mock_agent=use_mock_agent),
    ]
    _write_checks(run_dir, "benchmark_artifact_checks", checks)
    return checks
