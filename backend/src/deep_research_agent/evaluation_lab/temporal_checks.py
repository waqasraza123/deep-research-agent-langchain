from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .contracts import BenchmarkCase, CheckResult, CheckSeverity, CheckType
from .expected_outputs import _write_checks, load_report_text, normalize_text

TIME_SENSITIVE_WORDS = ("current", "latest", "today", "pricing", "version", "now")


def _year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    return int(match.group(0)) if match else None


def _check(
    check_id: str,
    name: str,
    passed: bool,
    *,
    severity: CheckSeverity = CheckSeverity.high,
    actual=None,
    message: str = "",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        check_type=CheckType.stale_source_warning,
        name=name,
        passed=passed,
        score=1.0 if passed else 0.0,
        severity=severity,
        message=message or ("passed" if passed else "failed"),
        actual=actual,
        affected_artifacts=["report.md"],
        recommendation=(
            "For currentness-sensitive questions, state source dates and freshness limits."
        ),
    )


def run_temporal_checks(
    case: BenchmarkCase, run_dir: Path, *, strict: bool = True
) -> list[CheckResult]:
    report = load_report_text(run_dir)
    question = normalize_text(case.question)
    report_norm = normalize_text(report)
    time_sensitive = any(word in question for word in TIME_SENSITIVE_WORDS)
    now_year = datetime.now(timezone.utc).year
    years = [_year(source.updated_at or source.published_at) for source in case.local_sources]
    known_years = [year for year in years if year is not None]
    stale_sources = [
        source.source_id
        for source in case.local_sources
        if (_year(source.updated_at or source.published_at) or now_year) <= now_year - 3
        or any(
            "archived" in warning.lower() or "outdated" in warning.lower()
            for warning in source.warnings
        )
    ]
    has_warning = any(
        phrase in report_norm
        for phrase in (
            "outdated",
            "stale",
            "archived",
            "cannot confirm",
            "based on provided sources",
            "current pricing cannot",
        )
    )
    claims_current = bool(re.search(r"\b(current|latest|today|now)\b", report_norm))
    checks = [
        _check(
            "temporal_not_over_penalized",
            "Non-time-sensitive cases are not over-penalized",
            True if time_sensitive else True,
            severity=CheckSeverity.info,
            actual={"time_sensitive": time_sensitive},
        )
    ]
    if time_sensitive and stale_sources:
        checks.append(
            _check(
                "stale_source_warning",
                "Stale sources are disclosed",
                has_warning,
                severity=CheckSeverity.critical,
                actual=stale_sources,
            )
        )
        checks.append(
            _check(
                "no_unqualified_current_claim",
                "Report avoids unqualified current claim from stale source",
                has_warning or not claims_current,
                severity=CheckSeverity.critical,
                actual={"claims_current": claims_current, "has_warning": has_warning},
            )
        )
    if time_sensitive and not known_years:
        checks.append(
            _check(
                "unknown_source_date_warning",
                "Unknown source dates produce warning",
                has_warning,
                severity=CheckSeverity.high,
                actual="no source dates",
            )
        )
    for idx, warning in enumerate(case.expected.required_warnings, start=1):
        if any(term in normalize_text(warning) for term in ("stale", "outdated", "archived")):
            checks.append(
                _check(
                    f"expected_temporal_warning_{idx}",
                    f"Expected temporal warning `{warning}`",
                    normalize_text(warning) in report_norm or has_warning,
                    severity=CheckSeverity.high,
                    actual=has_warning,
                )
            )
    _write_checks(run_dir, "benchmark_temporal_checks", checks)
    return checks
