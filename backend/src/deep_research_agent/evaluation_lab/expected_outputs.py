from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import BenchmarkCase, CheckResult, CheckSeverity, CheckType, model_to_plain


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def normalize_number(value: str) -> str:
    text = str(value).strip().lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _contains(text: str, phrase: str) -> bool:
    if phrase.startswith("re:"):
        return re.search(phrase[3:], text, flags=re.IGNORECASE | re.MULTILINE) is not None
    return normalize_text(phrase) in normalize_text(text)


def load_report_text(run_dir: Path) -> str:
    path = run_dir / "report.md"
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def load_notes_text(run_dir: Path) -> str:
    path = run_dir / "notes.md"
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def load_all_relevant_text(run_dir: Path) -> str:
    parts = [load_report_text(run_dir), load_notes_text(run_dir)]
    for rel in ("source_warnings.md", "contradictions.md", "unsupported_claims.md"):
        path = run_dir / rel
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def _result(
    check_id: str,
    check_type: CheckType,
    name: str,
    passed: bool,
    *,
    expected=None,
    actual=None,
    severity: CheckSeverity = CheckSeverity.medium,
    message: str = "",
    artifact: str = "report.md",
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
        affected_artifacts=[artifact],
        recommendation=recommendation,
    )


def run_expected_output_checks(case: BenchmarkCase, run_dir: Path) -> list[CheckResult]:
    report = load_report_text(run_dir)
    combined = load_all_relevant_text(run_dir)
    checks: list[CheckResult] = []

    for idx, phrase in enumerate(case.expected.must_mention, start=1):
        checks.append(
            _result(
                f"must_mention_{idx}",
                CheckType.must_mention,
                f"Must mention `{phrase}`",
                _contains(combined, phrase),
                expected=phrase,
                actual="present" if _contains(combined, phrase) else "missing",
                recommendation="Add the expected fact using only supplied source evidence.",
            )
        )
    for idx, phrase in enumerate(case.expected.must_not_mention, start=1):
        present = _contains(report, phrase)
        checks.append(
            _result(
                f"must_not_mention_{idx}",
                CheckType.must_not_mention,
                f"Must not mention `{phrase}`",
                not present,
                expected=f"absence of {phrase}",
                actual="present" if present else "absent",
                severity=CheckSeverity.high,
                recommendation="Remove or qualify forbidden wording.",
            )
        )
    for idx, entity in enumerate(case.expected.expected_entities, start=1):
        checks.append(
            _result(
                f"entity_{idx}",
                CheckType.entity_coverage,
                f"Expected entity `{entity}`",
                _contains(combined, entity),
                expected=entity,
                actual="present" if _contains(combined, entity) else "missing",
                recommendation="Cover all expected entities from the fixture sources.",
            )
        )
    for idx, number in enumerate(case.expected.expected_numbers, start=1):
        n = normalize_number(number)
        text = normalize_number(combined)
        checks.append(
            _result(
                f"number_{idx}",
                CheckType.numeric_support,
                f"Expected number `{number}`",
                n in text,
                expected=number,
                actual="present" if n in text else "missing",
                severity=CheckSeverity.high,
                recommendation="Preserve numeric values and units from the source.",
            )
        )
    for idx, date in enumerate(case.expected.expected_dates, start=1):
        checks.append(
            _result(
                f"date_{idx}",
                CheckType.date_support,
                f"Expected date `{date}`",
                _contains(combined, date),
                expected=date,
                actual="present" if _contains(combined, date) else "missing",
            )
        )
    for idx, claim in enumerate(case.expected.expected_claims, start=1):
        checks.append(
            _result(
                f"claim_{idx}",
                CheckType.report_completeness,
                f"Expected claim `{claim}`",
                _contains(combined, claim),
                expected=claim,
                actual="present" if _contains(combined, claim) else "missing",
            )
        )
    for idx, claim in enumerate(case.expected.forbidden_claims, start=1):
        present = _contains(report, claim)
        checks.append(
            _result(
                f"forbidden_claim_{idx}",
                CheckType.must_not_mention,
                f"Forbidden claim `{claim}`",
                not present,
                expected=f"absence of {claim}",
                actual="present" if present else "absent",
                severity=CheckSeverity.critical,
                recommendation="Remove unsupported or known-false claim.",
            )
        )
    for idx, phrase in enumerate(case.expected.required_uncertainty_phrases, start=1):
        checks.append(
            _result(
                f"uncertainty_{idx}",
                CheckType.uncertainty_handling,
                f"Required uncertainty `{phrase}`",
                _contains(combined, phrase),
                expected=phrase,
                actual="present" if _contains(combined, phrase) else "missing",
                severity=CheckSeverity.high,
                recommendation="State uncertainty, limitations, or conflicts explicitly.",
            )
        )
    for idx, warning in enumerate(
        [*case.expected.required_warnings, *case.expected.expected_missing_evidence_warnings],
        start=1,
    ):
        checks.append(
            _result(
                f"warning_{idx}",
                CheckType.uncertainty_handling,
                f"Required warning `{warning}`",
                _contains(combined, warning),
                expected=warning,
                actual="present" if _contains(combined, warning) else "missing",
                severity=CheckSeverity.high,
            )
        )
    citation_text = combined + "\n" + _sources_json_text(run_dir)
    for idx, source in enumerate(case.expected.expected_citation_sources, start=1):
        checks.append(
            _result(
                f"expected_citation_{idx}",
                CheckType.citation_support,
                f"Expected citation/source `{source}`",
                _contains(citation_text, source),
                expected=source,
                actual="present" if _contains(citation_text, source) else "missing",
                severity=CheckSeverity.high,
            )
        )

    _write_checks(run_dir, "benchmark_expected_output_checks", checks)
    return checks


def _sources_json_text(run_dir: Path) -> str:
    path = run_dir / "sources.json"
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _write_checks(run_dir: Path, stem: str, checks: list[CheckResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{stem}.json").write_text(
        json.dumps(model_to_plain(checks), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [f"# {stem}", ""]
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"- {mark} `{check.check_id}` {check.name}: {check.message}")
    (run_dir / f"{stem}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
