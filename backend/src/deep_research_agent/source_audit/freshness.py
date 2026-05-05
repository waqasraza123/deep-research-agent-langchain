from __future__ import annotations

import re
from datetime import date, datetime, timezone

from ._heuristics import clamp, text_head, url_path
from .contracts import SourceFreshnessScore

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

FULL_DATE_PATTERNS = (
    re.compile(r"\b(20\d{2}|19\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])(?=\D|$)"),
    re.compile(r"\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](20\d{2}|19\d{2})(?=\D|$)"),
    re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        r")\.?\s+([0-3]?\d),?\s+(20\d{2}|19\d{2})\b",
        re.IGNORECASE,
    ),
)
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")
VERSION_PATTERN = re.compile(r"\b(?:v|version|api|release)\s*([0-9]+(?:\.[0-9]+){1,3})\b", re.I)
FRESHNESS_KEYWORDS = (
    "latest",
    "current",
    "today",
    "this year",
    "2026",
    "pricing",
    "price",
    "api docs",
    "api reference",
    "legal",
    "law",
    "rules",
    "regulation",
    "regulations",
    "compliance",
    "benchmark",
    "benchmarks",
    "model",
    "models",
    "software version",
    "release",
    "changelog",
    "security",
)


def _parse_date_match(match: re.Match[str]) -> date | None:
    groups = match.groups()
    try:
        if len(groups) == 3 and groups[0].isdigit() and len(groups[0]) == 4:
            return date(int(groups[0]), int(groups[1]), int(groups[2]))
        if len(groups) == 3 and groups[2].isdigit() and len(groups[2]) == 4:
            first = groups[0]
            if first.isdigit():
                return date(int(groups[2]), int(groups[0]), int(groups[1]))
            return date(int(groups[2]), MONTHS[first.lower().rstrip(".")], int(groups[1]))
    except ValueError:
        return None
    return None


def freshness_matters(question: str) -> bool:
    q = (question or "").lower()
    return any(keyword in q for keyword in FRESHNESS_KEYWORDS)


def extract_dates(
    *,
    url: str = "",
    title: str | None = None,
    text: str = "",
    metadata: dict | None = None,
) -> list[date]:
    candidates: list[date] = []
    meta = metadata or {}
    meta_values = [
        meta.get("published_at"),
        meta.get("published"),
        meta.get("updated_at"),
        meta.get("last_modified"),
        meta.get("fetched_at"),
        meta.get("date"),
    ]
    chunks = [str(v) for v in meta_values if v]
    chunks.extend([title or "", url_path(url), text_head(text)])
    joined = "\n".join(chunks)

    for pattern in FULL_DATE_PATTERNS:
        for match in pattern.finditer(joined):
            parsed = _parse_date_match(match)
            if parsed is not None:
                candidates.append(parsed)

    for match in YEAR_PATTERN.finditer(joined):
        year = int(match.group(1))
        if 1990 <= year <= 2100:
            candidates.append(date(year, 1, 1))

    unique = sorted(set(candidates), reverse=True)
    return unique[:12]


def detect_versions(text: str, title: str | None = None, url: str = "") -> list[str]:
    joined = "\n".join([title or "", url, text_head(text, 20_000)])
    return list(dict.fromkeys(m.group(1) for m in VERSION_PATTERN.finditer(joined)))[:8]


def score_freshness(
    *,
    question: str,
    url: str,
    title: str | None,
    text: str,
    metadata: dict | None = None,
    now: datetime | None = None,
) -> SourceFreshnessScore:
    now_dt = now or datetime.now(timezone.utc)
    today = now_dt.date()
    matters = freshness_matters(question)
    dates = extract_dates(url=url, title=title, text=text, metadata=metadata)
    versions = detect_versions(text, title=title, url=url)
    reasons: list[str] = []

    if matters:
        reasons.append("Freshness matters for this question.")
    else:
        reasons.append("Freshness is not central to this question.")

    if versions:
        reasons.append("Version indicators found: " + ", ".join(versions[:3]) + ".")

    if not dates:
        score = 0.45 if not matters else 0.25
        if versions:
            score += 0.15
        return SourceFreshnessScore(
            score=round(clamp(score), 3),
            status="unknown",
            freshness_matters=matters,
            detected_dates=[],
            best_date=None,
            age_days=None,
            reasons=reasons + ["No reliable publication or update date was detected."],
        )

    best = dates[0]
    age_days = max(0, (today - best).days)
    if age_days <= 120:
        status = "current"
        score = 1.0
    elif age_days <= 540:
        status = "recent"
        score = 0.82
    elif age_days <= 1460:
        status = "possibly_stale"
        score = 0.58
    else:
        status = "stale"
        score = 0.28

    if not matters and status in {"possibly_stale", "stale"}:
        score += 0.12
    if matters and status in {"possibly_stale", "stale"}:
        score -= 0.12

    reasons.append(f"Best detected date is {best.isoformat()} ({age_days} days old).")
    return SourceFreshnessScore(
        score=round(clamp(score), 3),
        status=status,
        freshness_matters=matters,
        detected_dates=[d.isoformat() for d in dates],
        best_date=best.isoformat(),
        age_days=age_days,
        reasons=reasons,
    )
