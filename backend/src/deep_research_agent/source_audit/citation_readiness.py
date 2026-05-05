from __future__ import annotations

from ._heuristics import clamp, is_stable_url
from .contracts import (
    CitationReadiness,
    PrimarySourceAssessment,
    SourceAuditWarning,
    SourceAuthorityScore,
    SourceFreshnessScore,
)


def score_citation_readiness(
    *,
    url: str,
    title: str | None,
    word_count: int,
    freshness: SourceFreshnessScore,
    authority: SourceAuthorityScore,
    primary: PrimarySourceAssessment,
    warnings: list[SourceAuditWarning],
    duplicate_of: str | None = None,
) -> CitationReadiness:
    stable_url = is_stable_url(url)
    has_title = bool((title or "").strip())
    sufficient_content = word_count >= 160
    low_warning_count = len([w for w in warnings if w.severity in {"medium", "high"}]) <= 1
    low_duplication_risk = duplicate_of is None
    has_date_or_version = (not freshness.freshness_matters) or bool(
        freshness.best_date or any("Version indicators" in reason for reason in freshness.reasons)
    )

    reasons: list[str] = []
    blockers: list[str] = []
    score = 0.0

    checks = [
        (stable_url, 0.16, "Stable URL.", "URL appears unstable or tracking/search based."),
        (has_title, 0.14, "Identifiable title.", "Missing identifiable title."),
        (
            sufficient_content,
            0.18,
            "Sufficient extracted content.",
            "Insufficient extracted content.",
        ),
        (
            has_date_or_version,
            0.16,
            "Date or version is present when freshness matters.",
            "Date/version is missing for a freshness-sensitive question.",
        ),
        (
            low_warning_count,
            0.12,
            "Extraction warning count is low.",
            "Medium/high extraction warnings need review.",
        ),
        (
            low_duplication_risk,
            0.08,
            "No duplicate-source relationship detected.",
            "Duplicate-source relationship detected.",
        ),
    ]
    for ok, weight, good, bad in checks:
        if ok:
            score += weight
            reasons.append(good)
        else:
            blockers.append(bad)

    authority_weight = max(authority.score, primary.likelihood)
    score += 0.16 * authority_weight
    if authority_weight >= 0.65:
        reasons.append("Authority is reasonable for direct citation.")
    elif authority_weight >= 0.4:
        reasons.append("Authority is acceptable for background but may need corroboration.")
    else:
        blockers.append("Authority is weak for direct citation.")

    ready_score = round(clamp(score), 3)
    citation_ready = ready_score >= 0.72 and not any(
        blocker.startswith(("Missing", "Date/version", "Authority is weak")) for blocker in blockers
    )

    return CitationReadiness(
        score=ready_score,
        citation_ready=citation_ready,
        stable_url=stable_url,
        has_title=has_title,
        has_date_or_version=has_date_or_version,
        sufficient_content=sufficient_content,
        low_warning_count=low_warning_count,
        low_duplication_risk=low_duplication_risk,
        reasons=reasons,
        blockers=blockers,
    )
