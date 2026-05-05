from __future__ import annotations

import hashlib
import re

from .contracts import NumericClaim, NumericValue
from .metric_detector import detect_metric_name

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₨": "PKR"}
_CURRENCY_CODES = {"USD", "PKR", "SAR", "EUR", "GBP", "AED", "CAD", "AUD"}
_MAGNITUDES = {
    "k": ("thousand", 1_000.0),
    "m": ("million", 1_000_000.0),
    "b": ("billion", 1_000_000_000.0),
    "thousand": ("thousand", 1_000.0),
    "million": ("million", 1_000_000.0),
    "billion": ("billion", 1_000_000_000.0),
}
_UNITS = [
    "requests per second",
    "request/sec",
    "requests/sec",
    "per second",
    "per user per month",
    "per user/month",
    "seconds",
    "second",
    "secs",
    "sec",
    "ms",
    "milliseconds",
    "tokens",
    "token",
    "gb",
    "mb",
    "kb",
    "users",
    "dollars",
    "usd",
    "pkr",
    "sar",
    "stars",
    "issues",
    "points",
    "score",
]

_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){2,}(?:[-+][A-Za-z0-9.]+)?\b", re.I)
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)
_RANGE_RE = re.compile(
    r"(?<![\w.])(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
    r"(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>[A-Za-z/%]+(?:\s+per\s+[A-Za-z]+)?)?",
    re.I,
)
_RATIO_RE = re.compile(r"(?<![\w.])(?P<a>\d+(?:\.\d+)?)\s*:\s*(?P<b>\d+(?:\.\d+)?)(?![\w.])")
_PERCENT_RE = re.compile(r"(?<![\w.])(?P<num>\d[\d,]*(?:\.\d+)?)\s*%")
_CURRENCY_RE = re.compile(
    r"(?P<prefix>\$|€|£|₨|\bUSD\b|\bPKR\b|\bSAR\b|\bEUR\b|\bGBP\b)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mag>k|m|b|thousand|million|billion)?\s*"
    r"(?P<suffix>\bUSD\b|\bPKR\b|\bSAR\b|\bEUR\b|\bGBP\b|dollars?)?",
    re.I,
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<num>\d[\d,]*(?:\.\d+)?)(?P<mag>k|m|b|thousand|million|billion)?"
    r"(?:\s*(?P<unit>[A-Za-z/%]+(?:\s+per\s+[A-Za-z]+)?(?:\s+per\s+[A-Za-z]+)?))?",
    re.I,
)


def _parse_float(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _apply_magnitude(value: float | None, magnitude: str | None) -> tuple[float | None, str | None]:
    if value is None or not magnitude:
        return value, None
    key = magnitude.lower()
    label, factor = _MAGNITUDES.get(key, (None, 1.0))
    return value * factor, label


def _context(text: str, start: int, end: int, radius: int = 90) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - radius) : min(len(text), end + radius)]).strip()


def _unit_after(text: str, end: int) -> str | None:
    tail = re.sub(r"\s+", " ", text[end : min(len(text), end + 45)]).strip().lower()
    for unit in sorted(_UNITS, key=len, reverse=True):
        if tail.startswith(unit):
            return unit
    return None


def _currency_from_parts(prefix: str | None, suffix: str | None) -> str | None:
    for part in (prefix or "", suffix or ""):
        clean = part.strip()
        if not clean:
            continue
        if clean in _CURRENCY_SYMBOLS:
            return _CURRENCY_SYMBOLS[clean]
        upper = clean.upper()
        if upper in _CURRENCY_CODES:
            return upper
        if upper.startswith("DOLLAR"):
            return "USD"
    return None


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and span[1] > start for start, end in spans)


def _mk_value(
    *,
    raw_text: str,
    normalized_value: float | None,
    source_id: str | None,
    source_url: str | None,
    context: str,
    confidence_score: float,
    kind: str,
    start: int,
    end: int,
    unit: str | None = None,
    currency: str | None = None,
    percentage: bool = False,
    magnitude: str | None = None,
    range_start: float | None = None,
    range_end: float | None = None,
    ratio_left: float | None = None,
    ratio_right: float | None = None,
) -> NumericValue:
    metric_name = detect_metric_name(context, raw_text)
    return NumericValue(
        raw_text=raw_text,
        normalized_value=normalized_value,
        unit=unit,
        currency=currency,
        percentage=percentage,
        magnitude=magnitude,
        source_id=source_id,
        source_url=source_url,
        context=context,
        confidence_score=confidence_score,
        kind=kind,  # type: ignore[arg-type]
        range_start=range_start,
        range_end=range_end,
        ratio_left=ratio_left,
        ratio_right=ratio_right,
        metric_name=metric_name,
        start_char=start,
        end_char=end,
    )


def extract_numeric_values(
    text: str,
    *,
    source_id: str | None = None,
    source_url: str | None = None,
    include_dates: bool = True,
) -> list[NumericValue]:
    values: list[NumericValue] = []
    occupied: list[tuple[int, int]] = []

    for match in _VERSION_RE.finditer(text):
        raw = match.group(0)
        ctx = _context(text, match.start(), match.end())
        values.append(
            _mk_value(
                raw_text=raw,
                normalized_value=None,
                source_id=source_id,
                source_url=source_url,
                context=ctx,
                confidence_score=0.95,
                kind="version",
                start=match.start(),
                end=match.end(),
            )
        )
        occupied.append(match.span())

    if include_dates:
        for match in _DATE_RE.finditer(text):
            if _overlaps(match.span(), occupied):
                continue
            raw = match.group(0)
            year = None
            ym = re.search(r"\b(19|20)\d{2}\b", raw)
            if ym:
                year = float(ym.group(0))
            ctx = _context(text, match.start(), match.end())
            values.append(
                _mk_value(
                    raw_text=raw,
                    normalized_value=year,
                    source_id=source_id,
                    source_url=source_url,
                    context=ctx,
                    confidence_score=0.82,
                    kind="date",
                    start=match.start(),
                    end=match.end(),
                    unit="date",
                )
            )
            occupied.append(match.span())

    for match in _RANGE_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        a = _parse_float(match.group("a"))
        b = _parse_float(match.group("b"))
        unit = (match.group("unit") or _unit_after(text, match.end()) or "").strip().lower() or None
        raw = match.group(0)
        ctx = _context(text, match.start(), match.end())
        values.append(
            _mk_value(
                raw_text=raw,
                normalized_value=(a + b) / 2 if a is not None and b is not None else None,
                source_id=source_id,
                source_url=source_url,
                context=ctx,
                confidence_score=0.84,
                kind="range",
                start=match.start(),
                end=match.end(),
                unit=unit,
                range_start=a,
                range_end=b,
            )
        )
        occupied.append(match.span())

    for match in _RATIO_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        a = _parse_float(match.group("a"))
        b = _parse_float(match.group("b"))
        raw = match.group(0)
        ctx = _context(text, match.start(), match.end())
        values.append(
            _mk_value(
                raw_text=raw,
                normalized_value=a / b if a is not None and b not in (None, 0.0) else None,
                source_id=source_id,
                source_url=source_url,
                context=ctx,
                confidence_score=0.86,
                kind="ratio",
                start=match.start(),
                end=match.end(),
                unit="ratio",
                ratio_left=a,
                ratio_right=b,
            )
        )
        occupied.append(match.span())

    for match in _PERCENT_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        raw = match.group(0)
        num = _parse_float(match.group("num"))
        ctx = _context(text, match.start(), match.end())
        values.append(
            _mk_value(
                raw_text=raw,
                normalized_value=num,
                source_id=source_id,
                source_url=source_url,
                context=ctx,
                confidence_score=0.93,
                kind="percentage",
                start=match.start(),
                end=match.end(),
                unit="%",
                percentage=True,
            )
        )
        occupied.append(match.span())

    for match in _CURRENCY_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        currency = _currency_from_parts(match.group("prefix"), match.group("suffix"))
        if currency is None:
            continue
        raw = match.group(0).strip()
        num, mag = _apply_magnitude(_parse_float(match.group("num")), match.group("mag"))
        unit = _unit_after(text, match.end())
        ctx = _context(text, match.start(), match.end())
        values.append(
            _mk_value(
                raw_text=raw,
                normalized_value=num,
                source_id=source_id,
                source_url=source_url,
                context=ctx,
                confidence_score=0.91,
                kind="currency",
                start=match.start(),
                end=match.end(),
                unit=unit,
                currency=currency,
                magnitude=mag,
            )
        )
        occupied.append(match.span())

    for match in _NUMBER_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        raw = match.group(0).strip()
        if not raw or re.fullmatch(r"\d{4}", raw) and include_dates:
            continue
        unit = (match.group("unit") or _unit_after(text, match.end()) or "").strip().lower() or None
        if unit and unit not in _UNITS and not unit.startswith(("request", "per ")):
            unit = None
            raw = match.group("num") + (match.group("mag") or "")
        num, mag = _apply_magnitude(_parse_float(match.group("num")), match.group("mag"))
        ctx = _context(text, match.start(), match.end())
        metric = detect_metric_name(ctx, raw)
        kind = (
            "benchmark"
            if metric in {"benchmark", "score"} or unit in {"score", "points"}
            else "number"
        )
        values.append(
            _mk_value(
                raw_text=raw,
                normalized_value=num,
                source_id=source_id,
                source_url=source_url,
                context=ctx,
                confidence_score=0.72 if unit is None else 0.82,
                kind=kind,
                start=match.start(),
                end=match.end(),
                unit=unit,
                magnitude=mag,
            )
        )
        occupied.append(match.span())

    values.sort(key=lambda v: (v.start_char if v.start_char is not None else -1, v.raw_text))
    return values


def _sentences(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n{2,}", text):
        end = match.start()
        sentence = text[start:end].strip()
        if sentence:
            out.append((sentence, start, end))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append((tail, start, len(text)))
    return out


def _claim_id(origin: str, ref: str | None, text: str) -> str:
    digest = hashlib.sha1(f"{origin}|{ref or ''}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"qclaim-{digest}"


def extract_numeric_claims(
    text: str,
    *,
    origin: str = "source",
    origin_ref: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
) -> list[NumericClaim]:
    claims: list[NumericClaim] = []
    for sentence, start, _ in _sentences(text):
        values = extract_numeric_values(sentence, source_id=source_id, source_url=source_url)
        values = [value for value in values if value.kind != "version"]
        if not values:
            continue
        comparative = None
        comparative_match = re.search(
            r"\b(higher|more|greater|larger|faster|cheaper|lower|less|slower)\b",
            sentence,
            re.I,
        )
        if comparative_match:
            comparative = comparative_match.group(1).lower()
        metric = next((value.metric_name for value in values if value.metric_name), None)
        for value in values:
            if value.start_char is not None:
                value.start_char += start
            if value.end_char is not None:
                value.end_char += start
        claims.append(
            NumericClaim(
                claim_id=_claim_id(origin, origin_ref, sentence),
                text=sentence,
                origin=origin,  # type: ignore[arg-type]
                origin_ref=origin_ref,
                source_id=source_id,
                source_url=source_url,
                metric_name=metric,
                values=values,
                comparative_operator=comparative,
                confidence_score=max(value.confidence_score for value in values),
            )
        )
    return claims
