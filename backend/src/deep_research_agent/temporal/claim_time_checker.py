from __future__ import annotations

import re
from datetime import date

from .contracts import SourceTemporalMetadata, TimeSensitiveClaim
from .date_extractor import extract_dates_from_text
from .version_detector import detect_version_signals

TEMPORAL_LANGUAGE_RE = re.compile(
    r"\b(latest|current|currently|today|now|recent|newest|as of|updated|released|"
    r"deprecated|pricing|price|market|benchmark|regulation|law|policy|security advisory)\b",
    re.I,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SOURCE_REF_RE = re.compile(r"\[S([0-9]+)\]|\bS([0-9]+)\b")


def extract_time_sensitive_claims(
    *,
    report_text: str,
    notes_text: str,
    sources: list[SourceTemporalMetadata],
) -> list[TimeSensitiveClaim]:
    claims: list[TimeSensitiveClaim] = []
    by_id = {source.source_id: source for source in sources}
    source_dates = {
        source.source_id: [
            item.normalized_date
            for item in source.extracted_dates
            if item.normalized_date and item.date_type != "accessed"
        ]
        for source in sources
    }
    newest_source_date = max(
        (source.newest_date for source in sources if source.newest_date),
        default=None,
    )

    for origin, text in (("report", report_text), ("notes", notes_text)):
        for idx, sentence in enumerate(_sentences(text), start=1):
            dates = extract_dates_from_text(sentence, origin=f"{origin}_claim")
            version_signals = detect_version_signals(sentence)
            temporal_language = list(
                dict.fromkeys(
                    m.group(0).lower() for m in TEMPORAL_LANGUAGE_RE.finditer(sentence)
                )
            )
            versions = [signal.normalized_version or signal.raw_text for signal in version_signals]
            if not dates and not versions and not temporal_language:
                continue
            source_ids = _source_ids(sentence)
            claim = TimeSensitiveClaim(
                claim_id=f"TC-{len(claims) + 1:03d}",
                text=sentence,
                origin=origin,  # type: ignore[arg-type]
                origin_ref=f"{origin}:{idx}",
                source_ids=source_ids,
                detected_dates=dates,
                detected_versions=versions,
                temporal_language=temporal_language,
            )
            _assess_claim(claim, source_dates, by_id, newest_source_date)
            claims.append(claim)
    return claims


def _assess_claim(
    claim: TimeSensitiveClaim,
    source_dates: dict[str, list[str | None]],
    sources_by_id: dict[str, SourceTemporalMetadata],
    newest_source_date: str | None,
) -> None:
    reasons: list[str] = []
    support_dates: list[str] = []
    referenced_sources = [sources_by_id[sid] for sid in claim.source_ids if sid in sources_by_id]
    for source in referenced_sources:
        support_dates.extend(d for d in source_dates.get(source.source_id, []) if d)
    if not support_dates and not claim.source_ids:
        support_dates = [
            d
            for dates in source_dates.values()
            for d in dates
            if d
        ][:8]
        if support_dates:
            reasons.append("No explicit citation found; compared against run-level source dates.")

    claim.support_source_dates = sorted(set(support_dates), reverse=True)
    has_current_language = any(
        word in {"latest", "current", "currently", "today", "now", "recent", "newest"}
        for word in claim.temporal_language
    )
    stale_reference_sources = referenced_sources or list(sources_by_id.values())
    stale_reference = any(
        source.currentness_status in {"possibly_stale", "stale", "unknown"}
        for source in stale_reference_sources
    )

    if claim.detected_dates and newest_source_date:
        newest_date = date.fromisoformat(newest_source_date)
        future_claim_dates = [
            item.normalized_date
            for item in claim.detected_dates
            if item.normalized_date and date.fromisoformat(item.normalized_date) > newest_date
        ]
        if future_claim_dates:
            claim.status = "contradictory_dates"
            reasons.append(
                "Claim date is newer than the newest available source date: "
                + ", ".join(future_claim_dates[:3])
                + "."
            )
        elif support_dates:
            claim.status = "temporally_supported"
            reasons.append("Claim has date-sensitive wording and dated source support.")
        else:
            claim.status = "missing_date_support"
            reasons.append("Claim contains dates but no dated source support was found.")
    elif has_current_language and stale_reference:
        claim.status = "stale_source_risk"
        reasons.append(
            "Current/latest wording relies on stale, possibly stale, or undated sources."
        )
    elif has_current_language and not support_dates:
        claim.status = "missing_date_support"
        reasons.append("Current/latest wording lacks dated source support.")
    elif support_dates:
        claim.status = "temporally_weak" if has_current_language else "temporally_supported"
        reasons.append(
            "Dated source support exists, but deterministic checks cannot verify full "
            "temporal scope."
        )
    elif claim.detected_versions:
        claim.status = "temporally_weak"
        reasons.append("Version-sensitive claim detected; no contradictory source dates found.")
    else:
        claim.status = "unknown"
        reasons.append("Insufficient temporal evidence for this claim.")

    if not referenced_sources and claim.source_ids:
        reasons.append("Claim cites source IDs that are not present in sources.json.")
    claim.reasons = reasons
    score_by_status = {
        "temporally_supported": 0.82,
        "temporally_weak": 0.52,
        "stale_source_risk": 0.28,
        "missing_date_support": 0.22,
        "contradictory_dates": 0.12,
        "unknown": 0.2,
    }
    claim.confidence_score = score_by_status[claim.status]


def _sentences(text: str) -> list[str]:
    chunks = [chunk.strip(" -\t") for chunk in SENTENCE_RE.split(text or "")]
    return [chunk for chunk in chunks if 30 <= len(chunk) <= 800 and not chunk.startswith("#")]


def _source_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in SOURCE_REF_RE.finditer(text or ""):
        value = match.group(1) or match.group(2)
        if value:
            ids.append(f"S{value}")
    return list(dict.fromkeys(ids))
