from __future__ import annotations

import re
from pathlib import Path

from .contracts import BenchmarkCase, CheckResult, CheckSeverity, CheckType
from .expected_outputs import _write_checks, load_report_text, normalize_text
from .fixture_corpus import FixtureCorpus

NUMBER_RE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|ms|mb|gb|requests per minute|rpm|usd|\$)?\b", re.I
)
DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b")
ABSOLUTE_WORDS = {
    "always",
    "never",
    "guaranteed",
    "proven",
    "unquestionably",
    "definitely",
    "best",
    "only",
}


def _source_text(case: BenchmarkCase) -> str:
    corpus = FixtureCorpus([case])
    parts = []
    for url in case.urls:
        try:
            parts.append(corpus.resolve(url).text)
        except Exception:
            pass
    return "\n".join(parts)


def _numbers(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", item.group(0).lower().replace(",", "")).strip()
        for item in NUMBER_RE.finditer(text)
    }


def _dates(text: str) -> set[str]:
    return {item.group(0) for item in DATE_RE.finditer(text)}


def _check(
    check_id: str,
    name: str,
    passed: bool,
    *,
    message: str,
    severity: CheckSeverity = CheckSeverity.high,
    actual=None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        check_type=CheckType.report_completeness if passed else CheckType.source_traceability,
        name=name,
        passed=passed,
        score=1.0 if passed else 0.0,
        severity=severity,
        message=message,
        actual=actual,
        affected_artifacts=["report.md"],
        recommendation="Tie every factual value and citation back to supplied source text.",
    )


def run_hallucination_checks(case: BenchmarkCase, run_dir: Path) -> list[CheckResult]:
    report = load_report_text(run_dir)
    source_text = _source_text(case)
    expected_numbers = {n.lower().replace(",", "") for n in case.expected.expected_numbers}
    source_numbers = _numbers(source_text).union(expected_numbers)
    report_numbers = _numbers(report)
    unsupported_numbers = sorted(n for n in report_numbers if n not in source_numbers)
    source_dates = _dates(source_text).union(set(case.expected.expected_dates))
    unsupported_dates = sorted(d for d in _dates(report) if d not in source_dates)

    source_entities = set(case.expected.expected_entities)
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+)*\b", source_text):
        source_entities.add(match.group(0))
    invented_entities = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+)*\b", report):
        entity = match.group(0)
        if entity in {"Mock Research Report", "Summary", "Limitations", "Source Summaries"}:
            continue
        if entity not in source_entities and not entity.startswith("S"):
            invented_entities.append(entity)
    invented_entities = sorted(set(invented_entities))[:20]

    lowered = normalize_text(report)
    absolute_hits = sorted(
        word for word in ABSOLUTE_WORDS if re.search(rf"\b{re.escape(word)}\b", lowered)
    )
    known_urls = {source.url for source in case.local_sources}
    cited_urls = set(re.findall(r"https?://[^\s\])>]+|benchmark://[^\s\])>]+", report))
    invented_citations = sorted(url for url in cited_urls if url not in known_urls)

    checks = [
        _check(
            "unsupported_numbers",
            "Report numbers are source-supported",
            not unsupported_numbers,
            message="unsupported numbers found"
            if unsupported_numbers
            else "all report numbers are source-supported",
            actual=unsupported_numbers,
        ),
        _check(
            "unsupported_dates",
            "Report dates are source-supported",
            not unsupported_dates,
            message="unsupported dates found"
            if unsupported_dates
            else "all report dates are source-supported",
            actual=unsupported_dates,
        ),
        _check(
            "invented_entities",
            "Report avoids introduced entities",
            not invented_entities,
            message="possible introduced entities found"
            if invented_entities
            else "no introduced entities found",
            actual=invented_entities,
            severity=CheckSeverity.medium,
        ),
        _check(
            "absolute_language",
            "Report avoids unsupported absolute language",
            not absolute_hits,
            message="absolute language found" if absolute_hits else "no absolute language found",
            actual=absolute_hits,
            severity=CheckSeverity.medium,
        ),
        _check(
            "invented_citations",
            "Report avoids invented URLs or citations",
            not invented_citations,
            message="unknown cited URLs found"
            if invented_citations
            else "no unknown cited URLs found",
            actual=invented_citations,
        ),
    ]
    for idx, claim in enumerate(case.expected.forbidden_claims, start=1):
        present = normalize_text(claim) in lowered
        checks.append(
            _check(
                f"forbidden_claim_hallucination_{idx}",
                f"Known unsupported claim absent `{claim}`",
                not present,
                message="forbidden claim present" if present else "forbidden claim absent",
                severity=CheckSeverity.critical,
                actual=claim if present else None,
            )
        )
    _write_checks(run_dir, "benchmark_hallucination_checks", checks)
    return checks
