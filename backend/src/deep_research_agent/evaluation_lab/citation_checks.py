from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from .contracts import BenchmarkCase, CheckResult, CheckSeverity, CheckType
from .expected_outputs import _write_checks, load_report_text, normalize_text
from .fixture_corpus import FixtureCorpus


class CitationSupportLevel(str, Enum):
    explicit_citation = "explicit_citation"
    source_mentioned = "source_mentioned"
    artifact_supported = "artifact_supported"
    source_text_supported = "source_text_supported"
    unsupported = "unsupported"
    unknown = "unknown"


def _source_text(case: BenchmarkCase) -> str:
    corpus = FixtureCorpus([case])
    return "\n".join(corpus.resolve(url).text for url in case.urls if url in corpus._url_map)


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
        check_type=CheckType.citation_support,
        name=name,
        passed=passed,
        score=1.0 if passed else 0.0,
        severity=severity,
        message=message or ("passed" if passed else "failed"),
        actual=actual,
        affected_artifacts=["report.md", "sources.json"],
        recommendation="Reference supplied source IDs or URLs for material claims.",
    )


def run_citation_checks(
    case: BenchmarkCase, run_dir: Path, *, strict: bool = False
) -> list[CheckResult]:
    report = load_report_text(run_dir)
    sources_path = run_dir / "sources.json"
    sources_text = (
        sources_path.read_text(encoding="utf-8", errors="ignore") if sources_path.exists() else ""
    )
    source_text = _source_text(case)
    checks: list[CheckResult] = []
    for source in case.local_sources:
        present = source.source_id in sources_text or source.url in sources_text
        checks.append(
            _check(
                f"sources_json_{source.source_id}",
                f"`{source.source_id}` appears in sources.json",
                present,
                actual=present,
            )
        )

    for idx, claim in enumerate(case.expected.expected_claims, start=1):
        claim_present = normalize_text(claim) in normalize_text(report)
        source_supported = normalize_text(claim) in normalize_text(source_text)
        citation_present = bool(re.search(r"\[S\d+\]", report)) or any(
            source.source_id in report or source.title in report for source in case.local_sources
        )
        passed = source_supported or (claim_present and citation_present)
        level = (
            CitationSupportLevel.explicit_citation
            if citation_present and claim_present
            else CitationSupportLevel.source_text_supported
            if source_supported
            else CitationSupportLevel.unsupported
        )
        checks.append(
            _check(
                f"claim_support_{idx}",
                f"Expected claim is source-supported `{claim}`",
                passed,
                actual=level.value,
            )
        )

    unknown_urls = sorted(
        url
        for url in re.findall(r"https?://[^\s\])>]+|benchmark://[^\s\])>]+", report)
        if url not in {source.url for source in case.local_sources}
    )
    checks.append(
        _check(
            "unknown_citation_urls",
            "Report does not cite unknown URLs",
            not unknown_urls,
            actual=unknown_urls,
        )
    )

    if strict or case.scoring_profile.strict_citations:
        explicit = any(
            source.source_id in report or source.title in report or source.url in report
            for source in case.local_sources
        )
        checks.append(
            _check(
                "strict_explicit_citations",
                "Strict citation mode requires known source references",
                explicit,
                actual=explicit,
                severity=CheckSeverity.high,
            )
        )

    if case.category.value == "missing_primary_source":
        warning_text = normalize_text(report)
        warned = any(
            phrase in warning_text
            for phrase in ("primary source", "official", "legal review", "insufficient evidence")
        )
        checks.append(
            _check(
                "missing_primary_warning",
                "Missing primary source warning is present",
                warned,
                actual=warned,
                severity=CheckSeverity.critical,
            )
        )

    _write_checks(run_dir, "benchmark_citation_checks", checks)
    return checks
