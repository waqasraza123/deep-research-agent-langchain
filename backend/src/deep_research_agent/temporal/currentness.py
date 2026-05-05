from __future__ import annotations

from datetime import date, datetime, timezone

from .contracts import (
    CurrentnessAssessment,
    CurrentnessStatus,
    SourceTemporalMetadata,
    TemporalWarning,
)


def assess_source_currentness(
    source: SourceTemporalMetadata,
    *,
    freshness_required: bool,
    today: date | None = None,
) -> SourceTemporalMetadata:
    today = today or datetime.now(timezone.utc).date()
    reasons: list[str] = []
    warnings: list[str] = []
    status: CurrentnessStatus = "unknown"
    best_date = source.newest_date

    if not best_date:
        status = "unknown"
        reasons.append("No publication, update, access, or reliable mentioned date was detected.")
        if freshness_required:
            warnings.append("Freshness matters, but this source has no reliable date.")
    else:
        age_days = max(0, (today - date.fromisoformat(best_date)).days)
        reasons.append(f"Newest detected date is {best_date} ({age_days} days old).")
        if freshness_required:
            if age_days <= 120:
                status = "current"
            elif age_days <= 540:
                status = "probably_current"
            elif age_days <= 1460:
                status = "possibly_stale"
            else:
                status = "stale"
        else:
            if age_days <= 365:
                status = "current"
            elif age_days <= 1095:
                status = "probably_current"
            elif age_days <= 2190:
                status = "possibly_stale"
            else:
                status = "stale"

    outdated = [signal for signal in source.version_signals if signal.outdated_hint]
    current = [signal for signal in source.version_signals if signal.current_hint]
    if outdated:
        reasons.append(
            "Outdated/version-specific signal found: "
            + ", ".join(signal.signal_type for signal in outdated[:3])
            + "."
        )
        warnings.append("Source may be version-specific, deprecated, archived, or legacy.")
        if status in {"current", "probably_current"}:
            status = "possibly_stale"
        elif status == "unknown":
            status = "possibly_stale"
    elif current:
        reasons.append(
            "Current/stable documentation signal found: "
            + ", ".join(signal.signal_type for signal in current[:3])
            + "."
        )
        if status == "unknown":
            status = "probably_current"

    source.currentness_status = status
    source.currentness_reasons = reasons
    source.warnings = list(dict.fromkeys([*source.warnings, *warnings]))
    return source


def assess_run_currentness(
    *,
    thread_id: str,
    question: str,
    freshness_required: bool,
    freshness_signals: list[str],
    sources: list[SourceTemporalMetadata],
    generated_at: str,
) -> CurrentnessAssessment:
    dated = [source for source in sources if source.newest_date]
    newest = max((source.newest_date for source in dated if source.newest_date), default=None)
    oldest = min((source.oldest_date for source in dated if source.oldest_date), default=None)
    stale_sources = [
        _source_label(source)
        for source in sources
        if source.currentness_status in {"possibly_stale", "stale"}
    ]
    unknown_sources = [_source_label(source) for source in sources if not source.newest_date]
    statuses = [source.currentness_status for source in sources]
    warnings: list[TemporalWarning] = []
    reasons: list[str] = []

    if not sources:
        status: CurrentnessStatus = "unknown"
        reasons.append("No sources were available for temporal assessment.")
    elif freshness_required and unknown_sources and len(unknown_sources) == len(sources):
        status = "unknown"
        reasons.append("The question is time-sensitive and all sources have unknown dates.")
    elif freshness_required and any(item == "stale" for item in statuses):
        status = "stale"
        reasons.append("At least one source appears stale for a time-sensitive question.")
    elif freshness_required and (
        any(item == "possibly_stale" for item in statuses) or unknown_sources
    ):
        status = "possibly_stale"
        reasons.append("Freshness matters and at least one source is stale or undated.")
    elif any(item in {"current", "probably_current"} for item in statuses):
        status = "probably_current" if "probably_current" in statuses else "current"
        reasons.append("At least one source has a recent or current temporal signal.")
    else:
        status = "unknown"
        reasons.append("Temporal signals were insufficient for a strong currentness judgment.")

    if freshness_required and unknown_sources:
        warnings.append(
            TemporalWarning(
                warning_id="TW-unknown-source-dates",
                severity="high",
                category="unknown_source_dates",
                message="Time-sensitive research includes sources without reliable dates.",
                evidence=unknown_sources[:10],
            )
        )
    if stale_sources:
        warnings.append(
            TemporalWarning(
                warning_id="TW-stale-sources",
                severity="high" if freshness_required else "medium",
                category="stale_sources",
                message="One or more sources may be stale or version-specific.",
                evidence=stale_sources[:10],
            )
        )

    assessment = CurrentnessAssessment(
        thread_id=thread_id,
        question=question,
        generated_at=generated_at,
        status=status,
        freshness_required=freshness_required,
        freshness_signals=freshness_signals,
        source_count=len(sources),
        newest_source_date=newest,
        oldest_source_date=oldest,
        stale_sources=stale_sources,
        unknown_date_sources=unknown_sources,
        source_assessments=sources,
        reasons=reasons,
        warnings=warnings,
    )
    assessment.temporal_warning_block = render_temporal_warning_block(assessment, [])
    return assessment


def render_temporal_warning_block(
    assessment: CurrentnessAssessment,
    claim_warnings: list[str],
) -> str:
    lines = [
        "Temporal currentness check:",
        f"- Freshness matters: {'yes' if assessment.freshness_required else 'no'}",
        f"- Overall currentness: {assessment.status}",
        f"- Newest source date: {assessment.newest_source_date or 'unknown'}",
        f"- Oldest source date: {assessment.oldest_source_date or 'unknown'}",
        f"- Stale/version-risk sources: {len(assessment.stale_sources)}",
        f"- Unknown-date sources: {len(assessment.unknown_date_sources)}",
    ]
    if assessment.freshness_signals:
        lines.append("- Freshness signals: " + ", ".join(assessment.freshness_signals[:8]))
    if assessment.stale_sources:
        lines.append("- Stale sources: " + "; ".join(assessment.stale_sources[:5]))
    if assessment.unknown_date_sources:
        lines.append("- Unknown-date sources: " + "; ".join(assessment.unknown_date_sources[:5]))
    if claim_warnings:
        lines.append("- Date-sensitive claims needing caution: " + "; ".join(claim_warnings[:5]))
    return "\n".join(lines)


def _source_label(source: SourceTemporalMetadata) -> str:
    if source.source_url:
        return f"{source.source_id} {source.source_url}"
    return source.source_id
