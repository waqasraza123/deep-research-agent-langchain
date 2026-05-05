from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .artifacts import now_iso_utc
from .source_identity import ResearchWarning, WarningSeverity

ADVANCED_INTELLIGENCE_ARTIFACTS = (
    "advanced_intelligence_summary.json",
    "advanced_intelligence_summary.md",
)


class AdvancedIntelligenceSummary(BaseModel):
    thread_id: str
    generated_at: str
    question: str = ""
    overall_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence_level: str = "very_low"
    source_safety: dict[str, Any] = Field(default_factory=dict)
    temporal: dict[str, Any] = Field(default_factory=dict)
    quantitative: dict[str, Any] = Field(default_factory=dict)
    hypotheses: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ResearchWarning] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


def build_advanced_intelligence_summary(
    run_dir: Path,
    *,
    thread_id: str,
    question: str = "",
) -> AdvancedIntelligenceSummary:
    source_safety = _load_json_object(run_dir / "source_safety.json")
    currentness = _load_json_object(run_dir / "currentness_assessment.json")
    temporal_profile = _load_json_object(run_dir / "temporal_profile.json")
    quantitative = _load_json_object(run_dir / "quantitative_profile.json")
    hypotheses = _load_json_object(run_dir / "hypotheses.json")
    manifest = _load_json_object(run_dir / "artifact_manifest.json")
    reproducibility = _load_json_object(run_dir / "reproducibility_report.json")

    warnings: list[ResearchWarning] = []
    warnings.extend(_warnings_from_artifact("source_safety", source_safety))
    warnings.extend(_warnings_from_artifact("temporal", currentness))
    warnings.extend(_warnings_from_artifact("temporal", temporal_profile))
    warnings.extend(_warnings_from_artifact("quantitative", quantitative))

    hypothesis_summary = (
        hypotheses.get("summary") if isinstance(hypotheses.get("summary"), dict) else {}
    )
    warnings.extend(_hypothesis_warnings(hypothesis_summary))
    warnings.extend(_provenance_warnings(reproducibility))
    warnings = _dedupe_warnings(warnings)

    source_safety_summary = source_safety.get("summary") if source_safety else {}
    quantitative_summary = _quantitative_summary(quantitative)
    temporal_summary = _temporal_summary(currentness, temporal_profile)
    hypothesis_snapshot = _hypothesis_summary(hypothesis_summary)
    provenance_snapshot = _provenance_summary(manifest, reproducibility)

    confidence = _overall_confidence(
        warnings=warnings,
        hypothesis_average=_float_or_none(hypothesis_snapshot.get("average_confidence")),
        currentness_status=str(temporal_summary.get("currentness_status") or "unknown"),
        quantitative_failures=int(quantitative_summary.get("consistency_failures") or 0),
    )
    actions = _recommended_actions(warnings)
    return AdvancedIntelligenceSummary(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        question=question,
        overall_confidence_score=confidence,
        overall_confidence_level=_confidence_level(confidence),
        source_safety={
            "source_count": source_safety_summary.get("source_count", 0)
            if isinstance(source_safety_summary, dict)
            else 0,
            "high_risk_sources": source_safety_summary.get("high", 0)
            if isinstance(source_safety_summary, dict)
            else 0,
            "critical_risk_sources": source_safety_summary.get("critical", 0)
            if isinstance(source_safety_summary, dict)
            else 0,
            "excluded_from_agent_context": source_safety_summary.get(
                "excluded_from_agent_context", []
            )
            if isinstance(source_safety_summary, dict)
            else [],
        },
        temporal=temporal_summary,
        quantitative=quantitative_summary,
        hypotheses=hypothesis_snapshot,
        provenance=provenance_snapshot,
        warnings=warnings,
        recommended_actions=actions,
        artifact_paths={
            name: name
            for name in (
                "source_safety.json",
                "currentness_assessment.json",
                "quantitative_profile.json",
                "hypotheses.json",
                "artifact_manifest.json",
                "reproducibility_report.json",
            )
            if (run_dir / name).exists()
        },
    )


def write_advanced_intelligence_summary(
    run_dir: Path,
    summary: AdvancedIntelligenceSummary,
) -> list[str]:
    payload = _model_to_plain(summary)
    (run_dir / "advanced_intelligence_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "advanced_intelligence_summary.md").write_text(
        render_advanced_intelligence_summary_md(summary),
        encoding="utf-8",
    )
    return list(ADVANCED_INTELLIGENCE_ARTIFACTS)


def read_or_build_advanced_intelligence_summary(
    run_dir: Path,
    *,
    thread_id: str,
    question: str = "",
) -> AdvancedIntelligenceSummary:
    path = run_dir / "advanced_intelligence_summary.json"
    if not path.exists():
        summary = build_advanced_intelligence_summary(
            run_dir,
            thread_id=thread_id,
            question=question,
        )
        write_advanced_intelligence_summary(run_dir, summary)
        return summary
    data = json.loads(path.read_text(encoding="utf-8"))
    validator = getattr(AdvancedIntelligenceSummary, "model_validate", None)
    if callable(validator):
        return validator(data)
    return AdvancedIntelligenceSummary.parse_obj(data)


def render_advanced_intelligence_summary_md(summary: AdvancedIntelligenceSummary) -> str:
    lines = [
        "# Advanced Intelligence Summary",
        "",
        f"- Thread: `{summary.thread_id}`",
        f"- Generated: `{summary.generated_at}`",
        f"- Overall confidence: `{summary.overall_confidence_level}` "
        f"({summary.overall_confidence_score:.3f})",
        "",
        "## Source Safety",
        "",
        f"- Sources assessed: {summary.source_safety.get('source_count', 0)}",
        f"- High risk: {summary.source_safety.get('high_risk_sources', 0)}",
        f"- Critical risk: {summary.source_safety.get('critical_risk_sources', 0)}",
        f"- Excluded from agent context: "
        f"{', '.join(summary.source_safety.get('excluded_from_agent_context') or []) or 'none'}",
        "",
        "## Temporal",
        "",
        f"- Currentness: `{summary.temporal.get('currentness_status', 'unknown')}`",
        f"- Freshness required: {summary.temporal.get('freshness_required', False)}",
        f"- Stale sources: {summary.temporal.get('stale_source_count', 0)}",
        f"- Unknown-date sources: {summary.temporal.get('unknown_date_source_count', 0)}",
        "",
        "## Quantitative",
        "",
        f"- Numeric values: {summary.quantitative.get('value_count', 0)}",
        f"- Numeric claims: {summary.quantitative.get('claim_count', 0)}",
        f"- Tables: {summary.quantitative.get('table_count', 0)}",
        f"- CSVs: {summary.quantitative.get('csv_count', 0)}",
        f"- Consistency failures: {summary.quantitative.get('consistency_failures', 0)}",
        "",
        "## Hypotheses",
        "",
        f"- Total: {summary.hypotheses.get('total_hypotheses', 0)}",
        f"- Supported: {summary.hypotheses.get('supported', 0)}",
        f"- Contradicted: {summary.hypotheses.get('contradicted', 0)}",
        f"- Needs more evidence: {summary.hypotheses.get('needs_more_evidence', 0)}",
        f"- Average confidence: {summary.hypotheses.get('average_confidence', 0.0):.3f}",
        "",
        "## Warnings",
        "",
    ]
    if summary.warnings:
        for warning in summary.warnings:
            lines.append(
                f"- `{warning.severity.value}` `{warning.subsystem}` `{warning.code}`: "
                f"{warning.message}"
            )
    else:
        lines.append("- None.")
    if summary.recommended_actions:
        lines.extend(["", "## Recommended Actions", ""])
        lines.extend(f"- {action}" for action in summary.recommended_actions)
    return "\n".join(lines).rstrip() + "\n"


def _warnings_from_artifact(subsystem: str, data: dict[str, Any]) -> list[ResearchWarning]:
    raw = data.get("warnings") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    warnings: list[ResearchWarning] = []
    for idx, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            severity = _severity(item.get("severity") or item.get("risk_level"))
            warnings.append(
                ResearchWarning(
                    subsystem=str(item.get("subsystem") or subsystem),
                    code=str(item.get("code") or item.get("warning_id") or f"{subsystem}.warning"),
                    message=str(item.get("message") or item.get("explanation") or item),
                    severity=severity,
                    affected_artifacts=[
                        str(value) for value in item.get("affected_artifacts") or []
                    ],
                    affected_sources=[str(value) for value in item.get("affected_sources") or []],
                    recommended_action=str(item.get("recommended_action") or ""),
                )
            )
        else:
            warnings.append(
                ResearchWarning(
                    subsystem=subsystem,
                    code=f"{subsystem}.warning.{idx}",
                    message=str(item),
                    severity=WarningSeverity.MEDIUM,
                )
            )
    return warnings


def _hypothesis_warnings(summary: dict[str, Any]) -> list[ResearchWarning]:
    if not summary:
        return []
    warnings: list[ResearchWarning] = []
    for idx, message in enumerate(summary.get("warnings") or [], start=1):
        warnings.append(
            ResearchWarning(
                subsystem="hypotheses",
                code=f"hypotheses.summary_warning.{idx}",
                message=str(message),
                severity=WarningSeverity.MEDIUM,
                affected_artifacts=["hypotheses.json", "confidence_updates.json"],
                recommended_action="Review low-confidence or unresolved hypotheses.",
            )
        )
    if int(summary.get("contradicted") or 0) > 0:
        warnings.append(
            ResearchWarning(
                subsystem="hypotheses",
                code="hypotheses.contradicted",
                message="One or more hypotheses are contradicted by available evidence.",
                severity=WarningSeverity.HIGH,
                affected_artifacts=["hypotheses.json", "hypothesis_tests.json"],
                recommended_action="Resolve contradictions before treating conclusions as final.",
            )
        )
    return warnings


def _provenance_warnings(reproducibility: dict[str, Any]) -> list[ResearchWarning]:
    if not reproducibility:
        return []
    if reproducibility.get("can_replay") is True:
        return []
    return [
        ResearchWarning(
            subsystem="provenance",
            code="provenance.partial_replay",
            message="Run is not fully replayable from offline artifacts alone.",
            severity=WarningSeverity.LOW,
            affected_artifacts=["reproducibility_report.json", "replay_plan.json"],
            recommended_action="Use replay metadata to identify live sources or model credentials.",
        )
    ]


def _temporal_summary(
    currentness: dict[str, Any],
    temporal_profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "currentness_status": currentness.get("status", "unknown"),
        "freshness_required": bool(currentness.get("freshness_required")),
        "stale_source_count": len(currentness.get("stale_sources") or []),
        "unknown_date_source_count": len(currentness.get("unknown_date_sources") or []),
        "timeline_event_count": len(temporal_profile.get("timeline_events") or []),
        "warning_count": len(currentness.get("warnings") or [])
        + len(temporal_profile.get("warnings") or []),
    }


def _quantitative_summary(quantitative: dict[str, Any]) -> dict[str, Any]:
    return {
        "value_count": int(quantitative.get("value_count") or 0),
        "claim_count": int(quantitative.get("claim_count") or 0),
        "metric_count": int(quantitative.get("metric_count") or 0),
        "table_count": int(quantitative.get("table_count") or 0),
        "csv_count": int(quantitative.get("csv_count") or 0),
        "comparison_count": int(quantitative.get("comparison_count") or 0),
        "warning_count": int(quantitative.get("warning_count") or 0),
        "consistency_failures": int(quantitative.get("consistency_failures") or 0),
    }


def _hypothesis_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_hypotheses",
        "supported",
        "partially_supported",
        "contradicted",
        "unsupported",
        "inconclusive",
        "needs_more_evidence",
        "average_confidence",
    )
    return {key: summary.get(key, 0) for key in keys}


def _provenance_summary(
    manifest: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_count": len(manifest.get("artifacts") or []),
        "source_fingerprint_count": len(manifest.get("sources") or []),
        "reproducibility_status": reproducibility.get("status", "unknown"),
        "can_replay": bool(reproducibility.get("can_replay")),
    }


def _overall_confidence(
    *,
    warnings: list[ResearchWarning],
    hypothesis_average: float | None,
    currentness_status: str,
    quantitative_failures: int,
) -> float:
    score = 0.45
    if hypothesis_average is not None and hypothesis_average > 0:
        score = (score * 0.35) + (hypothesis_average * 0.65)
    severity_penalty = {
        WarningSeverity.INFO: 0.0,
        WarningSeverity.LOW: 0.015,
        WarningSeverity.MEDIUM: 0.04,
        WarningSeverity.HIGH: 0.09,
        WarningSeverity.CRITICAL: 0.16,
    }
    score -= min(0.35, sum(severity_penalty.get(w.severity, 0.04) for w in warnings))
    if currentness_status in {"stale", "possibly_stale", "unknown"}:
        score -= 0.06
    if quantitative_failures:
        score -= min(0.12, quantitative_failures * 0.04)
    return round(max(0.0, min(1.0, score)), 3)


def _confidence_level(score: float) -> str:
    if score >= 0.86:
        return "very_high"
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.30:
        return "low"
    return "very_low"


def _recommended_actions(warnings: list[ResearchWarning]) -> list[str]:
    actions = [
        warning.recommended_action
        for warning in warnings
        if warning.recommended_action
        and warning.severity
        in {
            WarningSeverity.MEDIUM,
            WarningSeverity.HIGH,
            WarningSeverity.CRITICAL,
        }
    ]
    return sorted(dict.fromkeys(actions))[:10]


def _dedupe_warnings(warnings: list[ResearchWarning]) -> list[ResearchWarning]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ResearchWarning] = []
    for warning in warnings:
        key = (warning.subsystem, warning.code, warning.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(warning)
    return out


def _severity(value: Any) -> WarningSeverity:
    normalized = str(value or "medium").lower()
    if normalized == "warning":
        normalized = "medium"
    if normalized == "error":
        normalized = "high"
    try:
        return WarningSeverity(normalized)
    except ValueError:
        return WarningSeverity.MEDIUM


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists() or path.is_dir():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _model_to_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [_model_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _model_to_plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value
