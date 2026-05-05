from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import urlparse

from .contracts import DatePrecision, DateType, ExtractedDate

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

MONTH_RE = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(19\d{2}|20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"),
        "ymd",
    ),
    (
        re.compile(
            rf"\b({MONTH_RE})\.?\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+(19\d{{2}}|20\d{{2}})\b",
            re.I,
        ),
        "mdy",
    ),
    (
        re.compile(
            rf"\b([0-3]?\d)(?:st|nd|rd|th)?\s+({MONTH_RE})\.?,?\s+(19\d{{2}}|20\d{{2}})\b",
            re.I,
        ),
        "dmy",
    ),
    (
        re.compile(rf"\b({MONTH_RE})\.?\s+(19\d{{2}}|20\d{{2}})\b", re.I),
        "my",
    ),
    (
        re.compile(r"\b(19\d{2}|20\d{2})\b"),
        "year",
    ),
)

QUESTION_FRESHNESS_SIGNALS: tuple[str, ...] = (
    "latest",
    "current",
    "currently",
    "today",
    "now",
    "recent",
    "newest",
    "up to date",
    "as of",
    "pricing",
    "price",
    "api docs",
    "api reference",
    "documentation",
    "regulation",
    "regulations",
    "law",
    "legal",
    "policy",
    "market",
    "benchmarks",
    "benchmark",
    "versions",
    "version",
    "releases",
    "release",
    "model capabilities",
    "capabilities",
    "job",
    "company",
    "security advisory",
    "security advisories",
    "cve",
)

CONTEXT_TYPES: tuple[tuple[str, DateType, float], ...] = (
    ("last modified", "updated", 0.9),
    ("last updated", "updated", 0.9),
    ("updated", "updated", 0.82),
    ("modified", "updated", 0.78),
    ("published", "published", 0.85),
    ("posted", "published", 0.75),
    ("accessed", "accessed", 0.82),
    ("retrieved", "accessed", 0.78),
    ("effective", "effective", 0.86),
    ("takes effect", "effective", 0.82),
    ("expires", "expired", 0.82),
    ("expired", "expired", 0.8),
    ("deadline", "deadline", 0.82),
    ("due by", "deadline", 0.78),
    ("release notes", "version_release", 0.84),
    ("release date", "version_release", 0.86),
    ("released", "version_release", 0.8),
    ("version", "version_release", 0.72),
    ("as of", "event_date", 0.72),
)

METADATA_DATE_KEYS: tuple[tuple[str, DateType, float], ...] = (
    ("published_at", "published", 0.96),
    ("published", "published", 0.95),
    ("publication_date", "published", 0.95),
    ("datepublished", "published", 0.95),
    ("updated_at", "updated", 0.96),
    ("updated", "updated", 0.94),
    ("modified", "updated", 0.9),
    ("last_modified", "updated", 0.94),
    ("datemodified", "updated", 0.94),
    ("fetched_at", "accessed", 0.86),
    ("accessed_at", "accessed", 0.9),
    ("effective_date", "effective", 0.94),
    ("expires", "expired", 0.9),
    ("expiry", "expired", 0.9),
    ("release_date", "version_release", 0.94),
)


def detect_time_sensitive_question(
    question: str,
    *,
    current_year: int | None = None,
) -> tuple[bool, list[str]]:
    q = " ".join((question or "").lower().split())
    signals: list[str] = []
    for signal in QUESTION_FRESHNESS_SIGNALS:
        if signal in q:
            signals.append(signal)
    for match in re.finditer(r"\b(19\d{2}|20\d{2})\b", question or ""):
        year = int(match.group(1))
        if current_year is None or year >= current_year - 10:
            signals.append(str(year))
    return bool(signals), list(dict.fromkeys(signals))


def extract_dates_from_source(
    *,
    source: dict[str, Any],
    text: str = "",
    metadata: dict[str, Any] | None = None,
    max_text_chars: int = 80_000,
) -> list[ExtractedDate]:
    source_id = str(source.get("source_id") or source.get("id") or "")
    source_url = str(source.get("final_url") or source.get("url") or "")
    title = str(source.get("title") or "")
    merged_metadata: dict[str, Any] = {}
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata.update({k: v for k, v in source.items() if k not in {"source_identity"}})
    out: list[ExtractedDate] = []
    out.extend(
        extract_dates_from_metadata(
            merged_metadata,
            source_id=source_id,
            source_url=source_url,
        )
    )
    if title:
        out.extend(
            extract_dates_from_text(
                title,
                source_id=source_id,
                source_url=source_url,
                origin="title",
            )
        )
    if source_url:
        out.extend(
            extract_dates_from_text(
                urlparse(source_url).path.replace("/", " "),
                source_id=source_id,
                source_url=source_url,
                origin="url",
            )
        )
    if text:
        out.extend(
            extract_dates_from_text(
                text[:max_text_chars],
                source_id=source_id,
                source_url=source_url,
                origin="text",
            )
        )
    return dedupe_dates(out)


def extract_dates_from_metadata(
    metadata: dict[str, Any],
    *,
    source_id: str | None = None,
    source_url: str | None = None,
) -> list[ExtractedDate]:
    out: list[ExtractedDate] = []
    flattened = _flatten_metadata(metadata)
    for key, value in flattened.items():
        if value is None or isinstance(value, (dict, list, tuple)):
            continue
        inferred_type, confidence = _date_type_from_metadata_key(key)
        if inferred_type is None:
            metadata_date_re = (
                r"(date|time|modified|published|updated|accessed|effective|expire|release)"
            )
            if not re.search(metadata_date_re, key, re.I):
                continue
            inferred_type = "unknown"
            confidence = 0.62
        text = str(value)
        extracted = extract_dates_from_text(
            text,
            source_id=source_id,
            source_url=source_url,
            origin=f"metadata:{key}",
            default_date_type=inferred_type,
            base_confidence=confidence,
        )
        out.extend(extracted)
    return out


def extract_dates_from_text(
    text: str,
    *,
    source_id: str | None = None,
    source_url: str | None = None,
    origin: str = "text",
    default_date_type: DateType | None = None,
    base_confidence: float = 0.68,
) -> list[ExtractedDate]:
    if not text:
        return []
    out: list[ExtractedDate] = []
    occupied: list[tuple[int, int]] = []
    for pattern, kind in DATE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue
            parsed = _parse_match(match, kind)
            if parsed is None:
                continue
            normalized, precision = parsed
            if not _reasonable_date(normalized):
                continue
            context = _context(text, start, end)
            date_type, confidence = _infer_date_type_near(
                text,
                start,
                end,
                default_date_type,
                base_confidence,
            )
            if origin == "url" and date_type == "unknown":
                date_type = "published"
                confidence = max(confidence, 0.62)
            out.append(
                ExtractedDate(
                    raw_text=match.group(0),
                    normalized_date=normalized,
                    date_type=date_type,
                    confidence_score=round(min(1.0, confidence), 3),
                    source_id=source_id,
                    source_url=source_url,
                    text_offset=start if origin == "text" else None,
                    surrounding_context=context,
                    precision=precision,
                    origin=origin,
                )
            )
            occupied.append((start, end))
    return dedupe_dates(out)


def dedupe_dates(dates: list[ExtractedDate]) -> list[ExtractedDate]:
    best: dict[tuple[str | None, str, str, str | None], ExtractedDate] = {}
    for item in dates:
        key = (item.normalized_date, item.raw_text.lower(), item.origin, item.source_id)
        current = best.get(key)
        if current is None or item.confidence_score > current.confidence_score:
            best[key] = item
    return sorted(
        best.values(),
        key=lambda item: (
            item.normalized_date or "0000-00-00",
            item.confidence_score,
            item.origin,
        ),
        reverse=True,
    )


def _flatten_metadata(metadata: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        clean_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metadata(value, clean_key))
        else:
            out[clean_key] = value
    return out


def _date_type_from_metadata_key(key: str) -> tuple[DateType | None, float]:
    normalized = key.lower().replace("-", "_")
    compact = normalized.replace("_", "").replace(".", "")
    for needle, date_type, confidence in METADATA_DATE_KEYS:
        if needle in normalized or needle.replace("_", "") in compact:
            return date_type, confidence
    return None, 0.58


def _parse_match(match: re.Match[str], kind: str) -> tuple[str, DatePrecision] | None:
    try:
        groups = match.groups()
        if kind == "ymd":
            d = date(int(groups[0]), int(groups[1]), int(groups[2]))
            return d.isoformat(), "day"
        if kind == "mdy":
            d = date(int(groups[2]), MONTHS[groups[0].lower().rstrip(".")], int(groups[1]))
            return d.isoformat(), "day"
        if kind == "dmy":
            d = date(int(groups[2]), MONTHS[groups[1].lower().rstrip(".")], int(groups[0]))
            return d.isoformat(), "day"
        if kind == "my":
            d = date(int(groups[1]), MONTHS[groups[0].lower().rstrip(".")], 1)
            return d.isoformat(), "month"
        if kind == "year":
            return f"{int(groups[0]):04d}-01-01", "year"
    except (KeyError, ValueError):
        return None
    return None


def _context(text: str, start: int, end: int, *, radius: int = 90) -> str:
    chunk = text[max(0, start - radius) : min(len(text), end + radius)]
    return " ".join(chunk.split())


def _infer_date_type_near(
    text: str,
    start: int,
    end: int,
    default_date_type: DateType | None,
    base_confidence: float,
) -> tuple[DateType, float]:
    before = re.split(r"[.!?\n]", text[max(0, start - 90) : start])[-1].lower()
    after = re.split(r"[.!?\n]", text[end : min(len(text), end + 45)])[0].lower()
    before_result = _infer_date_type(before, None, base_confidence)
    if before_result[0] != "mentioned_date":
        return before_result
    after_result = _infer_date_type(after, None, base_confidence)
    if after_result[0] != "mentioned_date":
        return after_result
    return _infer_date_type("", default_date_type, base_confidence)


def _infer_date_type(
    context: str,
    default_date_type: DateType | None,
    base_confidence: float,
) -> tuple[DateType, float]:
    lowered = context.lower()
    matches: list[tuple[int, DateType, float]] = []
    for needle, date_type, confidence in CONTEXT_TYPES:
        pos = lowered.rfind(needle)
        if pos >= 0:
            matches.append((pos, date_type, confidence))
    if matches:
        _, date_type, confidence = max(matches, key=lambda item: item[0])
        return date_type, max(base_confidence, confidence)
    if default_date_type:
        return default_date_type, base_confidence
    return "mentioned_date", min(base_confidence, 0.62)


def _reasonable_date(normalized: str) -> bool:
    year = int(normalized[:4])
    return 1900 <= year <= 2100


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < old_end and end > old_start for old_start, old_end in spans)
