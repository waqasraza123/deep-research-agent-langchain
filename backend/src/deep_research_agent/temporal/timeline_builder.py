from __future__ import annotations

from .contracts import SourceTemporalMetadata, TimelineEvent, TimeSensitiveClaim


def build_timeline(
    *,
    sources: list[SourceTemporalMetadata],
    claims: list[TimeSensitiveClaim],
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    counter = 1
    for source in sources:
        for item in source.extracted_dates:
            if not item.normalized_date:
                continue
            events.append(
                TimelineEvent(
                    event_id=f"TE-{counter:04d}",
                    date=item.normalized_date,
                    date_precision=item.precision,
                    event_type=item.date_type,
                    description=_source_date_description(source.source_id, item.date_type),
                    source_id=source.source_id,
                    source_url=source.source_url,
                    raw_text=item.raw_text,
                    confidence_score=item.confidence_score,
                    origin=item.origin,
                )
            )
            counter += 1
        for signal in source.version_signals:
            if not source.newest_date:
                continue
            timeline_signal_types = {
                "release_notes",
                "changelog",
                "semantic_version",
                "major_version",
            }
            if signal.signal_type not in timeline_signal_types:
                continue
            events.append(
                TimelineEvent(
                    event_id=f"TE-{counter:04d}",
                    date=source.newest_date,
                    date_precision="day",
                    event_type="version_signal",
                    description=f"{source.source_id} version signal: {signal.raw_text}",
                    source_id=source.source_id,
                    source_url=source.source_url,
                    raw_text=signal.raw_text,
                    confidence_score=signal.confidence_score,
                    origin="version_detector",
                )
            )
            counter += 1
    for claim in claims:
        for item in claim.detected_dates:
            if not item.normalized_date:
                continue
            events.append(
                TimelineEvent(
                    event_id=f"TE-{counter:04d}",
                    date=item.normalized_date,
                    date_precision=item.precision,
                    event_type="claim_date",
                    description=f"{claim.origin} claim date in {claim.claim_id}",
                    claim_id=claim.claim_id,
                    raw_text=item.raw_text,
                    confidence_score=item.confidence_score,
                    origin=claim.origin_ref or claim.origin,
                )
            )
            counter += 1
    return sorted(
        events,
        key=lambda item: (item.date, item.event_type, item.source_id or "", item.claim_id or ""),
    )


def _source_date_description(source_id: str, date_type: str) -> str:
    labels = {
        "published": "published",
        "updated": "updated",
        "accessed": "accessed/fetched",
        "effective": "became effective",
        "expired": "expired",
        "version_release": "version released",
        "deadline": "deadline",
        "event_date": "event date",
        "mentioned_date": "mentioned date",
        "unknown": "date",
    }
    return f"{source_id} {labels.get(date_type, date_type)}"
