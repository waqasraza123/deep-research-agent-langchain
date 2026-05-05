from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .contracts import FreshnessAssessment, Severity
from .coverage import is_time_sensitive

_LATEST_RE = re.compile(r"\b(latest|current|currently|recent|today|now|newest)\b", re.I)
_RESEARCH_DATE_RE = re.compile(r"\b(as of|research date|prepared on|updated on)\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_PRIMARY_NEED_RE = re.compile(
    r"\b(latest|current|official|regulation|legal|medical|clinical|release|pricing|api)\b", re.I
)


def assess_freshness(
    *,
    question: str,
    report_text: str,
    sources: list[dict[str, Any]],
    now: datetime | None = None,
) -> FreshnessAssessment:
    now = now or datetime.now(timezone.utc)
    sensitive = is_time_sensitive(question) or bool(_LATEST_RE.search(report_text))
    source_dates_present = any(_source_has_date(source) for source in sources)
    stale_sources = _stale_sources(sources, now)
    states_research_date = bool(_RESEARCH_DATE_RE.search(report_text))
    latest_wording_supported = bool(_LATEST_RE.search(report_text)) and source_dates_present
    primary_sources_needed = bool(_PRIMARY_NEED_RE.search(question)) and not _has_primary_hint(
        sources
    )
    reasons: list[str] = []

    if not sensitive:
        score = 0.8
        if not source_dates_present and sources:
            reasons.append("Source dates were not consistently present.")
    else:
        score = 0.2
        if source_dates_present:
            score += 0.22
        else:
            reasons.append("Time-sensitive research lacks source dates.")
        if states_research_date:
            score += 0.18
        else:
            reasons.append("Report does not state a research date.")
        if latest_wording_supported:
            score += 0.18
        elif _LATEST_RE.search(report_text):
            reasons.append("Latest/current wording is not backed by dated sources.")
        if not stale_sources:
            score += 0.14
        else:
            reasons.append("One or more sources appear stale for a current question.")
        if not primary_sources_needed:
            score += 0.08
        else:
            reasons.append("Primary or official current sources are needed.")

    score = round(max(0.0, min(1.0, score)), 3)
    return FreshnessAssessment(
        score=score,
        severity=_severity_for_score(score),
        is_time_sensitive=sensitive,
        source_dates_present=source_dates_present,
        stale_sources=stale_sources,
        states_research_date=states_research_date,
        latest_wording_supported=latest_wording_supported,
        primary_sources_needed=primary_sources_needed,
        reasons=reasons or ["Freshness checks did not detect major issues."],
    )


def _source_has_date(source: dict[str, Any]) -> bool:
    if source.get("fetched_at") or source.get("published_at") or source.get("updated_at"):
        return True
    text = " ".join(str(source.get(key) or "") for key in ("title", "summary", "date"))
    return bool(_YEAR_RE.search(text))


def _stale_sources(sources: list[dict[str, Any]], now: datetime) -> list[str]:
    stale: list[str] = []
    current_year = now.year
    for source in sources:
        years = [
            int(match.group(1))
            for match in _YEAR_RE.finditer(
                " ".join(str(source.get(key) or "") for key in ("title", "summary", "date"))
            )
        ]
        if years and max(years) <= current_year - 3:
            stale.append(str(source.get("url") or source.get("source_id") or source.get("id")))
    return stale


def _has_primary_hint(sources: list[dict[str, Any]]) -> bool:
    for source in sources:
        url = str(source.get("final_url") or source.get("url") or "").lower()
        title = str(source.get("title") or "").lower()
        if any(token in url for token in (".gov", ".edu", "docs.", "developer.", "sec.gov")):
            return True
        if any(token in title for token in ("official", "documentation", "release notes")):
            return True
    return False


def _severity_for_score(score: float) -> Severity:
    if score < 0.25:
        return "critical"
    if score < 0.45:
        return "high"
    if score < 0.65:
        return "medium"
    if score < 0.8:
        return "low"
    return "info"
