from __future__ import annotations

from ._heuristics import clamp, host_domain, url_path
from .authority import ACADEMIC_HOSTS, ACADEMIC_SUFFIXES, GOV_SUFFIXES, REPO_HOSTS, STANDARDS_HOSTS
from .contracts import PrimarySourceAssessment


def assess_primary_source(
    *,
    url: str,
    title: str | None,
    text: str,
    source_type: str = "unknown",
) -> PrimarySourceAssessment:
    domain = host_domain(url)
    path = url_path(url)
    joined = "\n".join([title or "", path, (text or "")[:60_000]]).lower()
    signals: list[str] = []
    likelihood = 0.2
    primary_type: str | None = None

    if any(domain == host or domain.endswith("." + host) for host in STANDARDS_HOSTS) or (
        "rfc" in path and domain.endswith("ietf.org")
    ):
        likelihood += 0.55
        primary_type = "standard_or_specification"
        signals.append("Standards body, RFC, or specification source.")
    if domain.endswith(GOV_SUFFIXES) or any(
        term in joined for term in ("federal register", "regulation", "statute", "official data")
    ):
        likelihood += 0.42
        primary_type = primary_type or "government_or_regulatory"
        signals.append("Government, legal, regulatory, or official dataset signal.")
    if domain.endswith(ACADEMIC_SUFFIXES) or domain in ACADEMIC_HOSTS or "doi:" in joined:
        likelihood += 0.35
        primary_type = primary_type or "academic_paper"
        signals.append("Academic paper or DOI signal.")
    if any(domain == host for host in REPO_HOSTS) and any(
        part in path for part in ("/releases", "/tags", "/blob/", "/tree/", "/commit/")
    ):
        likelihood += 0.55
        primary_type = primary_type or "repository_or_release"
        signals.append("Repository release, tag, commit, or source file.")
    if any(part in path for part in ("/docs", "/documentation", "/reference", "/api/")):
        likelihood += 0.28
        primary_type = primary_type or "official_documentation"
        signals.append("Documentation or reference material.")
    if any(part in path for part in ("/news/", "/press", "/announcements", "/blog/")) and any(
        phrase in joined for phrase in ("announces", "launched", "released", "we are")
    ):
        likelihood += 0.18
        primary_type = primary_type or "company_announcement"
        signals.append("Company announcement or original statement.")
    report_terms = ("original report", "survey methodology", "dataset", "white paper")
    if any(term in joined for term in report_terms):
        likelihood += 0.22
        primary_type = primary_type or "original_report"
        signals.append("Original report, dataset, or methodology language.")

    weak_secondary = (
        "affiliate" in joined
        or "/review" in path
        or "/tutorial" in path
        or "copied from" in joined
        or "originally published" in joined
    )
    if weak_secondary:
        likelihood -= 0.22
        signals.append("Secondary or copied-content language reduces primary-source likelihood.")

    score = round(clamp(likelihood), 3)
    if score >= 0.72:
        role = "primary"
    elif score >= 0.42:
        role = "secondary"
    elif score <= 0.25:
        role = "weak"
    else:
        role = "unknown"

    reasons = signals[:] or ["No clear primary-source indicators were detected."]
    return PrimarySourceAssessment(
        likelihood=score,
        source_role=role,  # type: ignore[arg-type]
        primary_type=primary_type,
        signals=signals,
        reasons=reasons,
    )
