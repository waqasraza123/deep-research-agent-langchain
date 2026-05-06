from __future__ import annotations

import re
from pathlib import Path

from .contracts import BenchmarkCase, CheckResult, CheckSeverity, CheckType
from .expected_outputs import _write_checks, load_report_text, normalize_number, normalize_text
from .fixture_corpus import FixtureCorpus
from .hallucination_checks import NUMBER_RE


def extract_numbers(text: str) -> set[str]:
    return {normalize_number(match.group(0)) for match in NUMBER_RE.finditer(text)}


def _check(
    check_id: str,
    name: str,
    passed: bool,
    *,
    severity: CheckSeverity = CheckSeverity.high,
    actual=None,
    expected=None,
    message: str = "",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        check_type=CheckType.numeric_support,
        name=name,
        passed=passed,
        score=1.0 if passed else 0.0,
        severity=severity,
        message=message or ("passed" if passed else "failed"),
        expected=expected,
        actual=actual,
        affected_artifacts=["report.md"],
        recommendation="Keep numeric values, units, and entity associations aligned with sources.",
    )


def _source_text(case: BenchmarkCase) -> str:
    corpus = FixtureCorpus([case])
    parts = []
    for url in case.urls:
        try:
            parts.append(corpus.resolve(url).text)
        except Exception:
            pass
    return "\n".join(parts)


def _entity_number_pairs(text: str) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for entity in re.findall(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)?\b", text):
        window_match = re.search(rf"{re.escape(entity)}[^.\n]{{0,80}}", text)
        if not window_match:
            continue
        nums = extract_numbers(window_match.group(0))
        if nums:
            pairs.setdefault(entity, set()).update(nums)
    return pairs


def run_numeric_checks(
    case: BenchmarkCase, run_dir: Path, *, strict: bool = True
) -> list[CheckResult]:
    report = load_report_text(run_dir)
    source_text = _source_text(case)
    source_numbers = extract_numbers(source_text)
    report_numbers = extract_numbers(report)
    expected_numbers = {normalize_number(item) for item in case.expected.expected_numbers}
    checks: list[CheckResult] = []
    for expected in expected_numbers:
        checks.append(
            _check(
                f"expected_number_{expected}",
                f"Expected number `{expected}` appears",
                expected in report_numbers,
                expected=expected,
                actual=sorted(report_numbers),
            )
        )
    unsupported = sorted(report_numbers - source_numbers - expected_numbers)
    checks.append(
        _check(
            "unsupported_numeric_values",
            "No unsupported numeric values",
            not unsupported,
            actual=unsupported,
        )
    )

    source_pairs = _entity_number_pairs(source_text)
    report_pairs = _entity_number_pairs(report)
    swapped = []
    for entity, nums in report_pairs.items():
        if entity in source_pairs and nums and not nums.issubset(source_pairs[entity]):
            swapped.append(
                {
                    "entity": entity,
                    "report_numbers": sorted(nums),
                    "source_numbers": sorted(source_pairs[entity]),
                }
            )
    checks.append(
        _check(
            "entity_number_association",
            "Numbers are not swapped between entities",
            not swapped,
            actual=swapped,
        )
    )

    units_missing = []
    for num in expected_numbers:
        if any(
            unit in num for unit in ("ms", "mb", "%", "requests per minute")
        ) and num not in normalize_number(report):
            units_missing.append(num)
    checks.append(
        _check(
            "unit_preservation",
            "Expected numeric units are preserved",
            not units_missing,
            actual=units_missing,
        )
    )

    report_norm = normalize_text(report)
    if any(trap.trap_type.value == "conflicting_number" for trap in case.traps):
        acknowledged = any(
            word in report_norm for word in ("conflict", "contradict", "different", "disagree")
        )
        checks.append(
            _check(
                "conflicting_numbers_acknowledged",
                "Conflicting numeric sources are acknowledged",
                acknowledged,
                severity=CheckSeverity.critical,
                actual=acknowledged,
            )
        )

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%\s+faster", report, flags=re.I):
        pct = float(match.group(1))
        nums = sorted(
            float(re.sub(r"[^\d.]", "", n)) for n in source_numbers if re.search(r"\d", n)
        )
        passed = True
        if len(nums) >= 2:
            smaller, larger = nums[0], nums[-1]
            actual_pct = (larger - smaller) / larger * 100
            passed = abs(pct - actual_pct) <= 2.0
        checks.append(
            _check(
                "percentage_calculation_tolerance",
                "Percentage calculations are within tolerance",
                passed,
                actual=pct,
            )
        )

    _write_checks(run_dir, "benchmark_numeric_checks", checks)
    return checks
