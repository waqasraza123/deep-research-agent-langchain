from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import REQUIRED_FILES, now_iso_utc, safe_thread_id

ReviewCriterionStatus = Literal["passed", "warning", "blocked", "not_applicable"]
ReviewDecision = Literal[
    "ready_for_approval",
    "changes_requested_recommended",
    "reject_recommended",
]

REVIEW_DOSSIER_JSON = "review_dossier.json"
REVIEW_DOSSIER_MD = "review_dossier.md"


class ReviewDossierRequest(BaseModel):
    reviewer: str | None = None
    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    require_export_bundle: bool = False
    require_replay_evidence: bool = False
    require_provenance: bool = True
    notes: str = ""


class ReviewCriterion(BaseModel):
    criterion_id: str
    title: str
    status: ReviewCriterionStatus
    severity: str = "medium"
    evidence_artifacts: list[str] = Field(default_factory=list)
    finding: str = ""
    required_action: str = ""


class ReviewDossier(BaseModel):
    dossier_version: str = "1.0"
    thread_id: str
    generated_at: str
    reviewer: str | None = None
    run_status: str = ""
    review_status: str = ""
    recommended_decision: ReviewDecision = "changes_requested_recommended"
    confidence_threshold: float = 0.55
    confidence_score: float | None = None
    confidence_level: str = "unknown"
    required_actions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    criteria: list[ReviewCriterion] = Field(default_factory=list)
    key_artifacts: dict[str, str] = Field(default_factory=dict)
    export_bundle: dict[str, Any] = Field(default_factory=dict)
    replay_evidence: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


def build_review_dossier(
    *,
    runs_dir: Path,
    thread_id: str,
    run: Any,
    request: ReviewDossierRequest | None = None,
) -> ReviewDossier:
    request = request or ReviewDossierRequest()
    run_dir = _safe_run_dir(runs_dir, thread_id)
    artifacts = _artifact_set(run_dir)
    advanced = _load_json(run_dir / "advanced_intelligence_summary.json")
    verification = _load_json(run_dir / "verification_report.json") or _load_json(
        run_dir / "verification_results.json"
    )
    quality = _load_json(run_dir / "quality_score.json")
    source_safety = _load_json(run_dir / "source_safety.json")
    currentness = _load_json(run_dir / "currentness_assessment.json")
    reproducibility = _load_json(run_dir / "reproducibility_report.json")
    export_manifest = _load_json(run_dir / "exports" / "export_manifest.json")
    replay_execution = _load_json(run_dir / "replay_execution.json")

    criteria = [
        _required_artifacts_criterion(artifacts),
        _run_errors_criterion(run),
        _confidence_criterion(advanced, request.confidence_threshold),
        _verification_criterion(verification),
        _quality_criterion(quality),
        _source_safety_criterion(source_safety),
        _currentness_criterion(currentness),
        _provenance_criterion(reproducibility, required=request.require_provenance),
        _export_criterion(export_manifest, required=request.require_export_bundle),
        _replay_criterion(replay_execution, required=request.require_replay_evidence),
    ]

    blockers = [criterion.finding for criterion in criteria if criterion.status == "blocked"]
    warnings = [
        criterion.finding
        for criterion in criteria
        if criterion.status == "warning" and criterion.finding
    ]
    run_warnings = [str(item) for item in getattr(run, "warnings", []) or []]
    warnings.extend(run_warnings[:20])
    actions = _required_actions(criteria)
    decision: ReviewDecision = "ready_for_approval"
    if any(c.status == "blocked" and c.severity == "critical" for c in criteria):
        decision = "reject_recommended"
    elif blockers or warnings:
        decision = "changes_requested_recommended"

    confidence_score = _float_or_none(advanced.get("overall_confidence_score"))
    dossier = ReviewDossier(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        reviewer=request.reviewer,
        run_status=str(getattr(getattr(run, "status", ""), "value", getattr(run, "status", ""))),
        review_status=str(
            getattr(getattr(getattr(run, "review", None), "status", ""), "value", "")
        ),
        recommended_decision=decision,
        confidence_threshold=request.confidence_threshold,
        confidence_score=confidence_score,
        confidence_level=str(advanced.get("overall_confidence_level") or "unknown"),
        required_actions=actions,
        blockers=blockers,
        warnings=_dedupe(warnings),
        criteria=criteria,
        key_artifacts=_key_artifacts(artifacts),
        export_bundle=_export_summary(export_manifest),
        replay_evidence=_replay_summary(replay_execution),
        notes=request.notes,
    )
    write_review_dossier(run_dir, dossier)
    return dossier


def read_review_dossier(runs_dir: Path, thread_id: str) -> ReviewDossier:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    path = run_dir / REVIEW_DOSSIER_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(ReviewDossier, "model_validate", None)
    if callable(validate):
        return validate(data)
    return ReviewDossier.parse_obj(data)


def write_review_dossier(run_dir: Path, dossier: ReviewDossier) -> list[str]:
    (run_dir / REVIEW_DOSSIER_JSON).write_text(
        json.dumps(_model_to_plain(dossier), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / REVIEW_DOSSIER_MD).write_text(
        render_review_dossier_markdown(dossier),
        encoding="utf-8",
    )
    return [REVIEW_DOSSIER_JSON, REVIEW_DOSSIER_MD]


def render_review_dossier_markdown(dossier: ReviewDossier) -> str:
    lines = [
        "# Review Dossier",
        "",
        f"- Thread ID: `{dossier.thread_id}`",
        f"- Generated at: `{dossier.generated_at}`",
        f"- Reviewer: {dossier.reviewer or 'unassigned'}",
        f"- Run status: `{dossier.run_status}`",
        f"- Review status: `{dossier.review_status or 'unknown'}`",
        f"- Recommended decision: `{dossier.recommended_decision}`",
        f"- Confidence: `{dossier.confidence_level}` "
        f"({dossier.confidence_score if dossier.confidence_score is not None else 'unknown'})",
        "",
        "## Criteria",
        "",
    ]
    for criterion in dossier.criteria:
        lines.extend(
            [
                f"### {criterion.title}",
                "",
                f"- Status: `{criterion.status}`",
                f"- Severity: `{criterion.severity}`",
                f"- Finding: {criterion.finding or 'None'}",
                f"- Required action: {criterion.required_action or 'None'}",
                "- Evidence: "
                + (", ".join(f"`{item}`" for item in criterion.evidence_artifacts) or "none"),
                "",
            ]
        )
    if dossier.blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {item}" for item in dossier.blockers)
        lines.append("")
    if dossier.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in dossier.warnings)
        lines.append("")
    if dossier.required_actions:
        lines.extend(["## Required Actions", ""])
        lines.extend(f"- {item}" for item in dossier.required_actions)
        lines.append("")
    if dossier.key_artifacts:
        lines.extend(["## Key Artifacts", ""])
        for label, path in sorted(dossier.key_artifacts.items()):
            lines.append(f"- {label}: `{path}`")
        lines.append("")
    if dossier.export_bundle:
        lines.extend(["## Export Bundle", ""])
        for key, value in dossier.export_bundle.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if dossier.replay_evidence:
        lines.extend(["## Replay Evidence", ""])
        for key, value in dossier.replay_evidence.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if dossier.notes:
        lines.extend(["## Notes", "", dossier.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _required_artifacts_criterion(artifacts: set[str]) -> ReviewCriterion:
    missing = [name for name in REQUIRED_FILES if name not in artifacts]
    if missing:
        return ReviewCriterion(
            criterion_id="required_artifacts",
            title="Required Deliverables",
            status="blocked",
            severity="critical",
            evidence_artifacts=list(REQUIRED_FILES),
            finding="Missing required artifacts: " + ", ".join(missing),
            required_action="Regenerate or backfill required deliverables before review.",
        )
    return ReviewCriterion(
        criterion_id="required_artifacts",
        title="Required Deliverables",
        status="passed",
        severity="low",
        evidence_artifacts=list(REQUIRED_FILES),
        finding="Required deliverables are present.",
    )


def _run_errors_criterion(run: Any) -> ReviewCriterion:
    errors = getattr(run, "errors", []) or []
    if errors:
        return ReviewCriterion(
            criterion_id="run_errors",
            title="Run Errors",
            status="blocked",
            severity="critical",
            finding=f"Run has {len(errors)} recorded error(s).",
            required_action="Resolve run errors or reject the output.",
        )
    return ReviewCriterion(
        criterion_id="run_errors",
        title="Run Errors",
        status="passed",
        severity="low",
        finding="No run errors are recorded.",
    )


def _confidence_criterion(data: dict[str, Any], threshold: float) -> ReviewCriterion:
    score = _float_or_none(data.get("overall_confidence_score"))
    evidence = ["advanced_intelligence_summary.json"] if data else []
    if score is None:
        return ReviewCriterion(
            criterion_id="confidence",
            title="Confidence",
            status="warning",
            severity="medium",
            evidence_artifacts=evidence,
            finding="Advanced confidence summary is unavailable.",
            required_action="Rebuild advanced intelligence summary or review manually.",
        )
    if score < threshold:
        return ReviewCriterion(
            criterion_id="confidence",
            title="Confidence",
            status="blocked",
            severity="high",
            evidence_artifacts=evidence,
            finding=f"Confidence {score:.3f} is below threshold {threshold:.3f}.",
            required_action="Request changes or add stronger evidence before approval.",
        )
    return ReviewCriterion(
        criterion_id="confidence",
        title="Confidence",
        status="passed",
        severity="low",
        evidence_artifacts=evidence,
        finding=f"Confidence {score:.3f} meets threshold {threshold:.3f}.",
    )


def _verification_criterion(data: dict[str, Any]) -> ReviewCriterion:
    evidence = ["verification_report.json"] if data else []
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else data
    unsupported = _int(summary.get("unsupported"))
    contradicted = _int(summary.get("contradicted"))
    open_issues = _int(summary.get("high_priority_open_issues"))
    if unsupported or contradicted or open_issues:
        return ReviewCriterion(
            criterion_id="verification",
            title="Verification",
            status="blocked",
            severity="high",
            evidence_artifacts=evidence,
            finding=(
                f"Verification found unsupported={unsupported}, contradicted={contradicted}, "
                f"high_priority_open_issues={open_issues}."
            ),
            required_action="Resolve verification issues or request changes.",
        )
    if not data:
        return ReviewCriterion(
            criterion_id="verification",
            title="Verification",
            status="warning",
            severity="medium",
            finding="Verification artifacts are unavailable.",
            required_action="Rebuild verification artifacts or perform manual claim review.",
        )
    return ReviewCriterion(
        criterion_id="verification",
        title="Verification",
        status="passed",
        severity="low",
        evidence_artifacts=evidence,
        finding="Verification summary has no high-priority unsupported or contradicted claims.",
    )


def _quality_criterion(data: dict[str, Any]) -> ReviewCriterion:
    score = _float_or_none(data.get("overall_score") or data.get("score"))
    if score is None:
        return ReviewCriterion(
            criterion_id="quality",
            title="Quality Score",
            status="warning",
            severity="medium",
            finding="Quality score is unavailable.",
            required_action="Rebuild evaluation artifacts or review quality manually.",
        )
    if score < 0.65:
        return ReviewCriterion(
            criterion_id="quality",
            title="Quality Score",
            status="blocked",
            severity="high",
            evidence_artifacts=["quality_score.json"],
            finding=f"Quality score {score:.3f} is below 0.650.",
            required_action="Improve coverage, citations, and report balance before approval.",
        )
    return ReviewCriterion(
        criterion_id="quality",
        title="Quality Score",
        status="passed",
        severity="low",
        evidence_artifacts=["quality_score.json"],
        finding=f"Quality score {score:.3f} is acceptable.",
    )


def _source_safety_criterion(data: dict[str, Any]) -> ReviewCriterion:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    critical = _int(summary.get("critical"))
    high = _int(summary.get("high"))
    if critical:
        return ReviewCriterion(
            criterion_id="source_safety",
            title="Source Safety",
            status="blocked",
            severity="critical",
            evidence_artifacts=["source_safety.json"],
            finding=f"Source safety found {critical} critical-risk source(s).",
            required_action="Reject or rerun without critical-risk source context.",
        )
    if high:
        return ReviewCriterion(
            criterion_id="source_safety",
            title="Source Safety",
            status="warning",
            severity="high",
            evidence_artifacts=["source_safety.json"],
            finding=f"Source safety found {high} high-risk source(s).",
            required_action="Confirm high-risk content was quoted/sanitized before approval.",
        )
    if not data:
        return ReviewCriterion(
            criterion_id="source_safety",
            title="Source Safety",
            status="warning",
            severity="medium",
            finding="Source safety artifacts are unavailable.",
            required_action="Rebuild source safety artifacts or manually inspect source text.",
        )
    return ReviewCriterion(
        criterion_id="source_safety",
        title="Source Safety",
        status="passed",
        severity="low",
        evidence_artifacts=["source_safety.json"],
        finding="No high or critical source safety findings are summarized.",
    )


def _currentness_criterion(data: dict[str, Any]) -> ReviewCriterion:
    status = str(data.get("status") or data.get("currentness_status") or "unknown")
    freshness_required = bool(data.get("freshness_required"))
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    if status in {"stale", "stale_source_risk", "contradictory_dates"}:
        return ReviewCriterion(
            criterion_id="currentness",
            title="Currentness",
            status="blocked",
            severity="high",
            evidence_artifacts=["currentness_assessment.json"],
            finding=f"Currentness status is {status}.",
            required_action="Refresh date-sensitive sources or constrain the claim timeframe.",
        )
    if freshness_required and (status == "unknown" or warnings):
        return ReviewCriterion(
            criterion_id="currentness",
            title="Currentness",
            status="warning",
            severity="medium",
            evidence_artifacts=["currentness_assessment.json"],
            finding=f"Freshness is required and currentness status is {status}.",
            required_action="Confirm source dates before approval.",
        )
    if not data:
        return ReviewCriterion(
            criterion_id="currentness",
            title="Currentness",
            status="not_applicable",
            severity="low",
            finding="Currentness artifacts are unavailable.",
        )
    return ReviewCriterion(
        criterion_id="currentness",
        title="Currentness",
        status="passed",
        severity="low",
        evidence_artifacts=["currentness_assessment.json"],
        finding=f"Currentness status is {status}.",
    )


def _provenance_criterion(data: dict[str, Any], *, required: bool) -> ReviewCriterion:
    if not data:
        return ReviewCriterion(
            criterion_id="provenance",
            title="Provenance",
            status="blocked" if required else "warning",
            severity="high" if required else "medium",
            finding="Reproducibility report is unavailable.",
            required_action="Refresh provenance artifacts before review.",
        )
    can_replay = bool(data.get("can_replay"))
    status = str(data.get("status") or "unknown")
    if not can_replay:
        return ReviewCriterion(
            criterion_id="provenance",
            title="Provenance",
            status="warning",
            severity="medium",
            evidence_artifacts=["reproducibility_report.json"],
            finding=f"Run is not fully replayable; reproducibility status is {status}.",
            required_action="Document live-source or model dependencies before approval.",
        )
    return ReviewCriterion(
        criterion_id="provenance",
        title="Provenance",
        status="passed",
        severity="low",
        evidence_artifacts=["reproducibility_report.json", "artifact_manifest.json"],
        finding=f"Reproducibility status is {status}.",
    )


def _export_criterion(data: dict[str, Any], *, required: bool) -> ReviewCriterion:
    if not data:
        return ReviewCriterion(
            criterion_id="export_bundle",
            title="Export Bundle",
            status="blocked" if required else "not_applicable",
            severity="medium",
            finding="Export bundle manifest is unavailable.",
            required_action="Generate a run export bundle when handoff packaging is required.",
        )
    return ReviewCriterion(
        criterion_id="export_bundle",
        title="Export Bundle",
        status="passed",
        severity="low",
        evidence_artifacts=["exports/export_manifest.json", "exports/run_export.zip"],
        finding=f"Export bundle is available with {data.get('exported_count', 0)} artifacts.",
    )


def _replay_criterion(data: dict[str, Any], *, required: bool) -> ReviewCriterion:
    if not data:
        return ReviewCriterion(
            criterion_id="replay_evidence",
            title="Replay Evidence",
            status="blocked" if required else "not_applicable",
            severity="medium",
            finding="Replay execution evidence is unavailable.",
            required_action="Run provenance replay when deterministic reproducibility evidence is required.",
        )
    mismatches = data.get("hash_mismatches") if isinstance(data.get("hash_mismatches"), dict) else {}
    missing = data.get("missing_expected_artifacts")
    missing_count = len(missing) if isinstance(missing, list) else 0
    if mismatches or missing_count:
        return ReviewCriterion(
            criterion_id="replay_evidence",
            title="Replay Evidence",
            status="warning",
            severity="medium",
            evidence_artifacts=["replay_execution.json"],
            finding=(
                f"Replay completed with {len(mismatches)} hash mismatch(es) and "
                f"{missing_count} missing expected artifact(s)."
            ),
            required_action="Inspect replay differences before approval.",
        )
    return ReviewCriterion(
        criterion_id="replay_evidence",
        title="Replay Evidence",
        status="passed",
        severity="low",
        evidence_artifacts=["replay_execution.json"],
        finding=f"Replay status is {data.get('status', 'unknown')}.",
    )


def _required_actions(criteria: list[ReviewCriterion]) -> list[str]:
    actions = [criterion.required_action for criterion in criteria if criterion.required_action]
    return _dedupe(actions)


def _key_artifacts(artifacts: set[str]) -> dict[str, str]:
    names = (
        "report.md",
        "advanced_intelligence_summary.md",
        "research_readiness.md",
        "verification_report.md",
        "quality_score.md",
        "source_safety.md",
        "currentness_assessment.md",
        "reproducibility_report.md",
        "exports/export_manifest.md",
        "replay_execution.md",
    )
    return {Path(name).stem: name for name in names if name in artifacts}


def _export_summary(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {}
    return {
        "archive_path": data.get("archive_path"),
        "archive_sha256": data.get("archive_sha256"),
        "exported_count": data.get("exported_count"),
        "skipped_count": data.get("skipped_count"),
        "profile": data.get("profile"),
    }


def _replay_summary(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {}
    mismatches = data.get("hash_mismatches") if isinstance(data.get("hash_mismatches"), dict) else {}
    missing = data.get("missing_expected_artifacts")
    return {
        "status": data.get("status"),
        "source_thread_id": data.get("source_thread_id"),
        "rebuilt_layers": len(data.get("rebuilt_layers") or []),
        "hash_mismatches": len(mismatches),
        "missing_expected_artifacts": len(missing) if isinstance(missing, list) else 0,
    }


def _artifact_set(run_dir: Path) -> set[str]:
    return {
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.is_dir():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    run_dir = (root / thread_id).resolve()
    if root != run_dir and root not in run_dir.parents:
        raise ValueError("Invalid thread_id")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(thread_id)
    return run_dir


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
