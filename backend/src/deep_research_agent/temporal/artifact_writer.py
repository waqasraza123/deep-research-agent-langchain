from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .claim_time_checker import extract_time_sensitive_claims
from .contracts import (
    CurrentnessAssessment,
    ExtractedDate,
    SourceTemporalMetadata,
    TemporalProfile,
    TemporalResearchSummary,
    TemporalWarning,
    TimelineEvent,
    TimeSensitiveClaim,
    model_to_plain,
)
from .currentness import (
    assess_run_currentness,
    assess_source_currentness,
    render_temporal_warning_block,
)
from .date_extractor import detect_time_sensitive_question, extract_dates_from_source
from .errors import TemporalArtifactError
from .timeline_builder import build_timeline
from .version_detector import detect_version_signals

TEMPORAL_ARTIFACTS = (
    "temporal_profile.json",
    "temporal_profile.md",
    "timeline.json",
    "timeline.md",
    "currentness_assessment.json",
    "currentness_assessment.md",
    "temporal_claims.json",
    "temporal_warnings.md",
)


@dataclass(frozen=True)
class TemporalArtifactBundle:
    profile: TemporalProfile
    timeline: list[TimelineEvent]
    currentness: CurrentnessAssessment
    claims: list[TimeSensitiveClaim]
    summary: TemporalResearchSummary


def rebuild_temporal_artifacts(
    run_dir: Path,
    *,
    thread_id: str,
    question: str = "",
    now: datetime | None = None,
    include_claims: bool = True,
) -> TemporalArtifactBundle:
    now = now or datetime.now(timezone.utc)
    generated_at = now.isoformat().replace("+00:00", "Z")
    sources_raw = _read_sources(run_dir / "sources.json")
    report_text = _read_text(run_dir / "report.md", max_chars=120_000) if include_claims else ""
    notes_text = _read_text(run_dir / "notes.md", max_chars=120_000) if include_claims else ""
    freshness_required, freshness_signals = detect_time_sensitive_question(
        question,
        current_year=now.year,
    )

    sources: list[SourceTemporalMetadata] = []
    all_dates: list[ExtractedDate] = []
    all_signals = []
    for idx, source in enumerate(sources_raw, start=1):
        metadata, text = _load_source_payload(run_dir, thread_id, source)
        source_id = str(source.get("source_id") or f"S{idx}")
        source_url = str(source.get("final_url") or source.get("url") or "")
        dates = extract_dates_from_source(source=source, text=text, metadata=metadata)
        for item in dates:
            if not item.source_id:
                item.source_id = source_id
            if not item.source_url:
                item.source_url = source_url
        signals = detect_version_signals(
            text,
            source_id=source_id,
            source_url=source_url,
            title=source.get("title"),
        )
        meta = SourceTemporalMetadata(
            source_id=source_id,
            source_url=source_url,
            title=source.get("title"),
            local_path=source.get("local_path"),
            fetched_at=source.get("fetched_at"),
            extracted_dates=dates,
            version_signals=signals,
        )
        _fill_source_date_rollups(meta)
        assess_source_currentness(meta, freshness_required=freshness_required, today=now.date())
        sources.append(meta)
        all_dates.extend(dates)
        all_signals.extend(signals)

    claims = (
        extract_time_sensitive_claims(
            report_text=report_text,
            notes_text=notes_text,
            sources=sources,
        )
        if include_claims
        else []
    )
    timeline = build_timeline(sources=sources, claims=claims)
    currentness = assess_run_currentness(
        thread_id=thread_id,
        question=question,
        freshness_required=freshness_required,
        freshness_signals=freshness_signals,
        sources=sources,
        generated_at=generated_at,
    )
    warnings = _merge_warnings(currentness.warnings, claims, sources, freshness_required)
    claim_warning_labels = [
        f"{claim.claim_id}: {claim.status}"
        for claim in claims
        if claim.status
        in {"stale_source_risk", "missing_date_support", "contradictory_dates", "unknown"}
    ]
    currentness.warnings = warnings
    currentness.temporal_warning_block = render_temporal_warning_block(
        currentness,
        claim_warning_labels,
    )
    profile = TemporalProfile(
        thread_id=thread_id,
        question=question,
        generated_at=generated_at,
        freshness_required=freshness_required,
        freshness_signals=freshness_signals,
        sources=sources,
        extracted_dates=all_dates,
        version_signals=all_signals,
        timeline_events=timeline,
        warnings=warnings,
        metadata={
            "include_claims": include_claims,
            "source_count": len(sources),
            "claim_count": len(claims),
        },
    )
    summary = TemporalResearchSummary(
        thread_id=thread_id,
        question=question,
        generated_at=generated_at,
        freshness_required=freshness_required,
        newest_source_date=currentness.newest_source_date,
        oldest_source_date=currentness.oldest_source_date,
        source_count=len(sources),
        stale_source_count=len(currentness.stale_sources),
        unknown_date_source_count=len(currentness.unknown_date_sources),
        date_sensitive_claim_count=len(claims),
        warning_count=len(warnings),
        currentness_status=currentness.status,
        temporal_warning_block=currentness.temporal_warning_block,
    )
    bundle = TemporalArtifactBundle(
        profile=profile,
        timeline=timeline,
        currentness=currentness,
        claims=claims,
        summary=summary,
    )
    write_temporal_artifacts(run_dir, bundle)
    return bundle


def write_temporal_artifacts(run_dir: Path, bundle: TemporalArtifactBundle) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "temporal_profile.json").write_text(
        json.dumps(model_to_plain(bundle.profile), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "temporal_profile.md").write_text(
        render_temporal_profile_md(bundle.profile, bundle.summary),
        encoding="utf-8",
    )
    (run_dir / "timeline.json").write_text(
        json.dumps(
            {
                "thread_id": bundle.profile.thread_id,
                "generated_at": bundle.profile.generated_at,
                "events": [model_to_plain(event) for event in bundle.timeline],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "timeline.md").write_text(render_timeline_md(bundle.timeline), encoding="utf-8")
    (run_dir / "currentness_assessment.json").write_text(
        json.dumps(model_to_plain(bundle.currentness), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "currentness_assessment.md").write_text(
        render_currentness_md(bundle.currentness),
        encoding="utf-8",
    )
    (run_dir / "temporal_claims.json").write_text(
        json.dumps(
            {
                "thread_id": bundle.profile.thread_id,
                "generated_at": bundle.profile.generated_at,
                "claims": [model_to_plain(claim) for claim in bundle.claims],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "temporal_warnings.md").write_text(
        render_warnings_md(bundle.currentness.warnings, bundle.summary),
        encoding="utf-8",
    )


def render_temporal_profile_md(
    profile: TemporalProfile,
    summary: TemporalResearchSummary,
) -> str:
    lines = [
        "# Temporal Profile",
        "",
        f"- Thread: `{profile.thread_id}`",
        f"- Generated: `{profile.generated_at}`",
        f"- Freshness required: {'yes' if profile.freshness_required else 'no'}",
        f"- Currentness: `{summary.currentness_status}`",
        f"- Sources: {summary.source_count}",
        f"- Extracted dates: {len(profile.extracted_dates)}",
        f"- Version signals: {len(profile.version_signals)}",
        f"- Date-sensitive claims: {summary.date_sensitive_claim_count}",
        "",
        "## Warning Block",
        "",
        "```text",
        summary.temporal_warning_block,
        "```",
        "",
        "## Sources",
        "",
    ]
    if not profile.sources:
        lines.append("- No sources available.")
    for source in profile.sources:
        lines.append(
            f"- `{source.source_id}` `{source.currentness_status}` "
            f"newest={source.newest_date or 'unknown'} title={source.title or source.source_url}"
        )
        for reason in source.currentness_reasons[:3]:
            lines.append(f"  - {reason}")
    return "\n".join(lines).rstrip() + "\n"


def render_timeline_md(events: list[TimelineEvent]) -> str:
    lines = ["# Timeline", ""]
    if not events:
        lines.append("No dated temporal events were detected.")
        return "\n".join(lines) + "\n"
    for event in events:
        subject = event.source_id or event.claim_id or "run"
        lines.append(
            f"- `{event.date}` `{event.event_type}` {subject}: {event.description} "
            f"(confidence {event.confidence_score:.2f})"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_currentness_md(assessment: CurrentnessAssessment) -> str:
    lines = [
        "# Currentness Assessment",
        "",
        f"- Status: `{assessment.status}`",
        f"- Freshness required: {'yes' if assessment.freshness_required else 'no'}",
        f"- Newest source date: `{assessment.newest_source_date or 'unknown'}`",
        f"- Oldest source date: `{assessment.oldest_source_date or 'unknown'}`",
        f"- Stale/version-risk sources: {len(assessment.stale_sources)}",
        f"- Unknown-date sources: {len(assessment.unknown_date_sources)}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in assessment.reasons)
    lines.extend(["", "## Source Assessments", ""])
    for source in assessment.source_assessments:
        lines.append(
            f"- `{source.source_id}` `{source.currentness_status}` "
            f"newest={source.newest_date or 'unknown'} url={source.source_url}"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_warnings_md(
    warnings: list[TemporalWarning],
    summary: TemporalResearchSummary,
) -> str:
    lines = [
        "# Temporal Warnings",
        "",
        "```text",
        summary.temporal_warning_block,
        "```",
        "",
    ]
    if not warnings:
        lines.append("No temporal warnings were generated.")
        return "\n".join(lines).rstrip() + "\n"
    for warning in warnings:
        lines.extend(
            [
                f"## {warning.warning_id}",
                "",
                f"- Severity: `{warning.severity}`",
                f"- Category: `{warning.category}`",
                f"- Message: {warning.message}",
            ]
        )
        if warning.source_id:
            lines.append(f"- Source: `{warning.source_id}` {warning.source_url or ''}".rstrip())
        if warning.claim_id:
            lines.append(f"- Claim: `{warning.claim_id}`")
        if warning.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {item}" for item in warning.evidence[:8])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _read_sources(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TemporalArtifactError(f"Invalid sources.json: {e}") from e
    if not isinstance(loaded, list):
        raise TemporalArtifactError("sources.json must contain a JSON list")
    return [item for item in loaded if isinstance(item, dict)]


def _load_source_payload(
    run_dir: Path,
    thread_id: str,
    source: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    metadata = dict(source)
    text = ""
    local_path = source.get("local_path")
    source_path = _safe_source_path(run_dir, thread_id, str(local_path or ""))
    if source_path is not None and source_path.exists() and not source_path.is_dir():
        text = _read_text(source_path, max_chars=160_000)
        meta_path = source_path.with_suffix(".json")
        if meta_path.exists() and not meta_path.is_dir():
            try:
                sibling = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(sibling, dict):
                    metadata.update(sibling)
            except Exception:
                pass
    return metadata, text


def _safe_source_path(run_dir: Path, thread_id: str, local_path: str) -> Path | None:
    if not local_path or local_path.startswith("/") or "\\" in local_path or ".." in local_path:
        return None
    rel = local_path
    marker = f"runs/{thread_id}/"
    if marker in rel:
        rel = rel.split(marker, 1)[-1]
    if rel.startswith("/") or "\\" in rel or ".." in rel:
        return None
    root = run_dir.resolve()
    candidate = (root / rel).resolve()
    if not str(candidate).startswith(str(root)):
        return None
    return candidate


def _read_text(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return text[:max_chars]


def _fill_source_date_rollups(source: SourceTemporalMetadata) -> None:
    content_dates = [
        item.normalized_date
        for item in source.extracted_dates
        if item.normalized_date and item.date_type not in {"accessed"}
    ]
    publication_dates = [
        item.normalized_date
        for item in source.extracted_dates
        if item.normalized_date and item.date_type == "published"
    ]
    update_dates = [
        item.normalized_date
        for item in source.extracted_dates
        if item.normalized_date and item.date_type == "updated"
    ]
    source.best_publication_date = max(publication_dates, default=None)
    source.best_update_date = max(update_dates, default=None)
    source.newest_date = max(content_dates, default=None)
    source.oldest_date = min(content_dates, default=None)


def _merge_warnings(
    base_warnings: list[TemporalWarning],
    claims: list[TimeSensitiveClaim],
    sources: list[SourceTemporalMetadata],
    freshness_required: bool,
) -> list[TemporalWarning]:
    warnings = list(base_warnings)
    for source in sources:
        if not source.warnings:
            continue
        warnings.append(
            TemporalWarning(
                warning_id=f"TW-source-{source.source_id}",
                severity="high" if freshness_required else "medium",
                category="source_currentness",
                message="; ".join(source.warnings[:3]),
                source_id=source.source_id,
                source_url=source.source_url,
                evidence=source.currentness_reasons[:5],
            )
        )
    for claim in claims:
        if claim.status not in {
            "stale_source_risk",
            "missing_date_support",
            "contradictory_dates",
            "unknown",
        }:
            continue
        severity = "critical" if claim.status == "contradictory_dates" else "high"
        warnings.append(
            TemporalWarning(
                warning_id=f"TW-claim-{claim.claim_id}",
                severity=severity,
                category="claim_temporal_support",
                message=f"Date-sensitive claim is {claim.status}.",
                claim_id=claim.claim_id,
                evidence=[claim.text, *claim.reasons],
            )
        )
    return _dedupe_warnings(warnings)


def _dedupe_warnings(warnings: list[TemporalWarning]) -> list[TemporalWarning]:
    seen: set[tuple[str, str | None, str | None]] = set()
    out: list[TemporalWarning] = []
    for warning in warnings:
        key = (warning.category, warning.source_id, warning.claim_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(warning)
    return out
