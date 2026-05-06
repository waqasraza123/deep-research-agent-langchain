"""Deterministic run custody certificates for audit handoff review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import REQUIRED_FILES, list_artifacts, now_iso_utc, safe_thread_id
from deep_research_agent.provenance.lineage import file_sha256

from .contracts import ResearchRun
from .export_bundle import export_bundle_path, read_export_manifest
from .operator_audit import verify_operator_audit
from .retention import read_retention_policy

CustodyCheckStatus = Literal["passed", "warning", "blocked"]
CustodyReadiness = Literal["ready", "needs_attention", "blocked"]

CUSTODY_CERTIFICATE_JSON = "custody_certificate.json"
CUSTODY_CERTIFICATE_MD = "custody_certificate.md"
DYNAMIC_ARTIFACTS = {
    CUSTODY_CERTIFICATE_JSON,
    CUSTODY_CERTIFICATE_MD,
    "operator_audit.jsonl",
    "operator_audit.md",
    "integrity_report.json",
    "integrity_report.md",
    "disclosure_report.json",
    "disclosure_report.md",
    "handoff_manifest.json",
    "handoff_manifest.md",
}


class RunCustodyRequest(BaseModel):
    requested_by: str = "operator"
    require_review_approval: bool = False
    require_export_bundle: bool = False
    require_retention_policy: bool = False
    require_operator_audit: bool = False
    require_provenance: bool = True
    include_artifact_hashes: bool = True
    max_artifacts: int = Field(default=1000, ge=1, le=10000)
    notes: str = ""


class CustodyCheck(BaseModel):
    check_id: str
    title: str
    status: CustodyCheckStatus
    finding: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustodyArtifactRecord(BaseModel):
    path: str
    size_bytes: int
    sha256: str


class CustodyArtifactInventory(BaseModel):
    artifact_count: int = 0
    hashed_count: int = 0
    total_bytes: int = 0
    truncated: bool = False
    excluded_dynamic_artifacts: list[str] = Field(default_factory=list)
    artifacts: list[CustodyArtifactRecord] = Field(default_factory=list)


class RunCustodyCertificate(BaseModel):
    certificate_version: str = "1.0"
    thread_id: str
    generated_at: str
    requested_by: str = "operator"
    readiness: CustodyReadiness = "needs_attention"
    checks: list[CustodyCheck] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_inventory: CustodyArtifactInventory = Field(default_factory=CustodyArtifactInventory)
    notes: str = ""


def build_run_custody_certificate(
    *,
    runs_dir: Path,
    thread_id: str,
    run: ResearchRun | None = None,
    request: RunCustodyRequest | None = None,
) -> RunCustodyCertificate:
    request = request or RunCustodyRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    run_dir = _safe_run_dir(runs_dir, thread_id)
    artifact_paths = _artifact_path_set(runs_dir, thread_id)
    checks = [
        _required_artifacts_check(artifact_paths),
        _run_state_check(run),
        _review_check(run, required=request.require_review_approval),
        _provenance_check(artifact_paths, required=request.require_provenance),
        _retention_check(runs_dir, thread_id, required=request.require_retention_policy),
        _export_check(runs_dir, thread_id, required=request.require_export_bundle),
        _operator_audit_check(runs_dir, thread_id, required=request.require_operator_audit),
    ]
    artifact_inventory = (
        _artifact_inventory(
            runs_dir=runs_dir,
            thread_id=thread_id,
            max_artifacts=request.max_artifacts,
        )
        if request.include_artifact_hashes
        else CustodyArtifactInventory(
            artifact_count=len(
                [path for path in artifact_paths if path not in DYNAMIC_ARTIFACTS]
            ),
            excluded_dynamic_artifacts=sorted(path for path in artifact_paths if path in DYNAMIC_ARTIFACTS),
        )
    )
    if artifact_inventory.truncated:
        checks.append(
            CustodyCheck(
                check_id="artifact_inventory_limit",
                title="Artifact Inventory Limit",
                status="warning",
                finding="Artifact inventory was truncated by max_artifacts.",
                required_action="Increase max_artifacts and regenerate the certificate for full custody hashing.",
                metadata={"max_artifacts": request.max_artifacts},
            )
        )
    blockers = [check.finding for check in checks if check.status == "blocked" and check.finding]
    warnings = [check.finding for check in checks if check.status == "warning" and check.finding]
    readiness: CustodyReadiness = "ready"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "needs_attention"
    certificate = RunCustodyCertificate(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        run_summary=_run_summary(run),
        artifact_inventory=artifact_inventory,
        notes=request.notes,
    )
    write_run_custody_certificate(run_dir, certificate)
    return certificate


def read_run_custody_certificate(runs_dir: Path, thread_id: str) -> RunCustodyCertificate:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    path = run_dir / CUSTODY_CERTIFICATE_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(RunCustodyCertificate, "model_validate", None)
    if callable(validate):
        return validate(data)
    return RunCustodyCertificate.parse_obj(data)


def write_run_custody_certificate(
    run_dir: Path,
    certificate: RunCustodyCertificate,
) -> list[str]:
    (run_dir / CUSTODY_CERTIFICATE_JSON).write_text(
        json.dumps(_model_to_plain(certificate), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / CUSTODY_CERTIFICATE_MD).write_text(
        render_run_custody_certificate_markdown(certificate),
        encoding="utf-8",
    )
    return [CUSTODY_CERTIFICATE_JSON, CUSTODY_CERTIFICATE_MD]


def render_run_custody_certificate_markdown(certificate: RunCustodyCertificate) -> str:
    lines = [
        "# Run Custody Certificate",
        "",
        f"- Thread ID: `{certificate.thread_id}`",
        f"- Generated at: `{certificate.generated_at}`",
        f"- Requested by: `{certificate.requested_by}`",
        f"- Readiness: `{certificate.readiness}`",
        f"- Blockers: {len(certificate.blockers)}",
        f"- Warnings: {len(certificate.warnings)}",
        "",
        "## Checks",
        "",
    ]
    for check in certificate.checks:
        lines.extend(
            [
                f"### {check.title}",
                "",
                f"- Status: `{check.status}`",
                f"- Finding: {check.finding or 'None'}",
                f"- Required action: {check.required_action or 'None'}",
                "- Evidence: "
                + (", ".join(f"`{item}`" for item in check.evidence_artifacts) or "none"),
                "",
            ]
        )
    if certificate.blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {item}" for item in certificate.blockers)
        lines.append("")
    if certificate.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in certificate.warnings)
        lines.append("")
    lines.extend(
        [
            "## Run Summary",
            "",
            f"- Status: `{certificate.run_summary.get('status', 'unknown')}`",
            f"- Review status: `{certificate.run_summary.get('review_status', 'unknown')}`",
            f"- Error count: {certificate.run_summary.get('error_count', 0)}",
            f"- Warning count: {certificate.run_summary.get('warning_count', 0)}",
            "",
            "## Artifact Inventory",
            "",
            f"- Artifact count: {certificate.artifact_inventory.artifact_count}",
            f"- Hashed count: {certificate.artifact_inventory.hashed_count}",
            f"- Total bytes: {certificate.artifact_inventory.total_bytes}",
            f"- Truncated: `{certificate.artifact_inventory.truncated}`",
        ]
    )
    if certificate.artifact_inventory.excluded_dynamic_artifacts:
        excluded = ", ".join(f"`{item}`" for item in certificate.artifact_inventory.excluded_dynamic_artifacts)
        lines.append(f"- Excluded dynamic artifacts: {excluded}")
    lines.append("")
    if certificate.artifact_inventory.artifacts:
        lines.extend(["### Artifact Hashes", ""])
        for artifact in certificate.artifact_inventory.artifacts:
            lines.append(
                f"- `{artifact.path}` ({artifact.size_bytes} bytes) sha256=`{artifact.sha256}`"
            )
        lines.append("")
    if certificate.notes:
        lines.extend(["## Notes", "", certificate.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _required_artifacts_check(artifact_paths: set[str]) -> CustodyCheck:
    missing = [path for path in REQUIRED_FILES if path not in artifact_paths]
    if missing:
        return CustodyCheck(
            check_id="required_artifacts",
            title="Required Artifacts",
            status="blocked",
            finding=f"Missing required artifacts: {', '.join(missing)}.",
            required_action="Regenerate or repair the run before custody handoff.",
            evidence_artifacts=[path for path in REQUIRED_FILES if path in artifact_paths],
            metadata={"missing": missing},
        )
    return CustodyCheck(
        check_id="required_artifacts",
        title="Required Artifacts",
        status="passed",
        finding="All required run artifacts are present.",
        evidence_artifacts=list(REQUIRED_FILES),
    )


def _run_state_check(run: ResearchRun | None) -> CustodyCheck:
    if run is None:
        return CustodyCheck(
            check_id="run_state",
            title="Run State",
            status="warning",
            finding="Run metadata was not found; custody is based only on artifact files.",
            required_action="Restore .run.json if lifecycle state is required for handoff.",
        )
    status = _enum_value(run.status)
    if status in {"failed", "cancelled"}:
        return CustodyCheck(
            check_id="run_state",
            title="Run State",
            status="blocked",
            finding=f"Run is in terminal error state `{status}`.",
            required_action="Resolve or explicitly document the failed/cancelled state before handoff.",
            metadata={"status": status},
        )
    if status == "waiting_for_review":
        return CustodyCheck(
            check_id="run_state",
            title="Run State",
            status="warning",
            finding="Run is waiting for review.",
            required_action="Complete review approval or document why custody proceeds before approval.",
            metadata={"status": status},
        )
    return CustodyCheck(
        check_id="run_state",
        title="Run State",
        status="passed",
        finding=f"Run state is `{status}`.",
        metadata={"status": status},
    )


def _review_check(run: ResearchRun | None, *, required: bool) -> CustodyCheck:
    review_status = "unknown"
    reviewer = None
    if run is not None:
        review_status = _enum_value(run.review.status)
        reviewer = run.review.reviewer
    evidence = ["review_dossier.json", "review_dossier.md"]
    if review_status == "approved":
        return CustodyCheck(
            check_id="review_approval",
            title="Review Approval",
            status="passed",
            finding="Run review is approved.",
            evidence_artifacts=evidence,
            metadata={"review_status": review_status, "reviewer": reviewer},
        )
    if review_status == "rejected":
        return CustodyCheck(
            check_id="review_approval",
            title="Review Approval",
            status="blocked",
            finding="Run review was rejected.",
            required_action="Do not use this run for approved custody handoff.",
            evidence_artifacts=evidence,
            metadata={"review_status": review_status, "reviewer": reviewer},
        )
    if required:
        return CustodyCheck(
            check_id="review_approval",
            title="Review Approval",
            status="blocked",
            finding=f"Required review approval is missing; current status is `{review_status}`.",
            required_action="Approve the run or regenerate custody without requiring review approval.",
            evidence_artifacts=evidence,
            metadata={"review_status": review_status, "reviewer": reviewer},
        )
    return CustodyCheck(
        check_id="review_approval",
        title="Review Approval",
        status="warning",
        finding=f"Review approval is not present; current status is `{review_status}`.",
        required_action="Use review endpoints if this handoff requires human approval.",
        evidence_artifacts=evidence,
        metadata={"review_status": review_status, "reviewer": reviewer},
    )


def _provenance_check(artifact_paths: set[str], *, required: bool) -> CustodyCheck:
    expected = [
        "artifact_manifest.json",
        "artifact_dependency_dag.json",
        "reproducibility_report.json",
        "replay_plan.json",
    ]
    missing = [path for path in expected if path not in artifact_paths]
    status: CustodyCheckStatus = "passed"
    if missing and required:
        status = "blocked"
    elif missing:
        status = "warning"
    finding = "Provenance artifacts are present."
    action = ""
    if missing:
        finding = f"Missing provenance artifacts: {', '.join(missing)}."
        action = "Refresh provenance artifacts before custody handoff."
    return CustodyCheck(
        check_id="provenance",
        title="Provenance",
        status=status,
        finding=finding,
        required_action=action,
        evidence_artifacts=[path for path in expected if path in artifact_paths],
        metadata={"missing": missing, "required": required},
    )


def _retention_check(runs_dir: Path, thread_id: str, *, required: bool) -> CustodyCheck:
    try:
        policy = read_retention_policy(runs_dir, thread_id)
    except FileNotFoundError:
        return CustodyCheck(
            check_id="retention_policy",
            title="Retention Policy",
            status="blocked" if required else "warning",
            finding="Retention policy is missing.",
            required_action="Set a retention policy if this run must be protected after handoff.",
            metadata={"required": required},
        )
    active_holds = [hold.hold_id for hold in policy.active_holds]
    return CustodyCheck(
        check_id="retention_policy",
        title="Retention Policy",
        status="passed",
        finding=f"Retention policy `{policy.retention_class}` is present.",
        evidence_artifacts=["retention_policy.json", "retention_policy.md"],
        metadata={
            "retention_class": policy.retention_class,
            "retain_until": policy.retain_until,
            "delete_after": policy.delete_after,
            "legal_hold": policy.legal_hold,
            "active_hold_ids": active_holds,
        },
    )


def _export_check(runs_dir: Path, thread_id: str, *, required: bool) -> CustodyCheck:
    try:
        manifest = read_export_manifest(runs_dir, thread_id)
        archive_path = export_bundle_path(runs_dir, thread_id)
    except FileNotFoundError:
        return CustodyCheck(
            check_id="export_bundle",
            title="Export Bundle",
            status="blocked" if required else "warning",
            finding="Export bundle is missing.",
            required_action="Create an export bundle if custody handoff requires a portable package.",
            metadata={"required": required},
        )
    archive_sha256 = file_sha256(archive_path)
    if manifest.archive_sha256 and archive_sha256 != manifest.archive_sha256:
        return CustodyCheck(
            check_id="export_bundle",
            title="Export Bundle",
            status="blocked",
            finding="Export archive hash does not match export manifest.",
            required_action="Regenerate the export bundle before handoff.",
            evidence_artifacts=[
                "exports/run_export.zip",
                "exports/export_manifest.json",
                "exports/export_manifest.md",
            ],
            metadata={
                "manifest_archive_sha256": manifest.archive_sha256,
                "actual_archive_sha256": archive_sha256,
            },
        )
    return CustodyCheck(
        check_id="export_bundle",
        title="Export Bundle",
        status="passed",
        finding="Export bundle exists and matches its manifest hash.",
        evidence_artifacts=[
            "exports/run_export.zip",
            "exports/export_manifest.json",
            "exports/export_manifest.md",
        ],
        metadata={
            "profile": manifest.profile,
            "archive_sha256": archive_sha256,
            "exported_count": manifest.exported_count,
            "skipped_count": manifest.skipped_count,
        },
    )


def _operator_audit_check(runs_dir: Path, thread_id: str, *, required: bool) -> CustodyCheck:
    global_verification = verify_operator_audit(runs_dir)
    run_verification = verify_operator_audit(runs_dir, thread_id)
    evidence = ["operator_audit.jsonl", "operator_audit.md"]
    metadata = {
        "required": required,
        "global": _model_to_plain(global_verification),
        "run": _model_to_plain(run_verification),
    }
    if not global_verification.valid or not run_verification.valid:
        return CustodyCheck(
            check_id="operator_audit",
            title="Operator Audit",
            status="blocked",
            finding="Operator audit verification failed.",
            required_action="Investigate audit log corruption before custody handoff.",
            evidence_artifacts=evidence,
            metadata=metadata,
        )
    if required and run_verification.event_count == 0:
        return CustodyCheck(
            check_id="operator_audit",
            title="Operator Audit",
            status="blocked",
            finding="Required per-run operator audit events are missing.",
            required_action="Perform required operator actions before custody handoff.",
            evidence_artifacts=evidence,
            metadata=metadata,
        )
    if run_verification.event_count == 0:
        return CustodyCheck(
            check_id="operator_audit",
            title="Operator Audit",
            status="warning",
            finding="No per-run operator audit events are present.",
            required_action="Generate review, export, retention, or custody audit events if needed.",
            evidence_artifacts=evidence,
            metadata=metadata,
        )
    return CustodyCheck(
        check_id="operator_audit",
        title="Operator Audit",
        status="passed",
        finding="Global and per-run operator audit chains verify.",
        evidence_artifacts=evidence,
        metadata=metadata,
    )


def _artifact_inventory(
    *,
    runs_dir: Path,
    thread_id: str,
    max_artifacts: int,
) -> CustodyArtifactInventory:
    thread_dir = _safe_run_dir(runs_dir, thread_id)
    artifacts = [
        artifact
        for artifact in list_artifacts(runs_dir, thread_id)
        if artifact.path not in DYNAMIC_ARTIFACTS
    ]
    total_bytes = sum(artifact.size_bytes for artifact in artifacts)
    selected = artifacts[:max_artifacts]
    records = [
        CustodyArtifactRecord(
            path=artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=file_sha256(thread_dir / artifact.path),
        )
        for artifact in selected
    ]
    return CustodyArtifactInventory(
        artifact_count=len(artifacts),
        hashed_count=len(records),
        total_bytes=total_bytes,
        truncated=len(artifacts) > len(selected),
        excluded_dynamic_artifacts=sorted(
            artifact.path
            for artifact in list_artifacts(runs_dir, thread_id)
            if artifact.path in DYNAMIC_ARTIFACTS
        ),
        artifacts=records,
    )


def _artifact_path_set(runs_dir: Path, thread_id: str) -> set[str]:
    return {artifact.path for artifact in list_artifacts(runs_dir, thread_id)}


def _run_summary(run: ResearchRun | None) -> dict[str, Any]:
    if run is None:
        return {"status": "unknown", "review_status": "unknown"}
    return {
        "status": _enum_value(run.status),
        "stage": _enum_value(run.current_stage),
        "review_status": _enum_value(run.review.status),
        "reviewer": run.review.reviewer,
        "error_count": len(run.errors),
        "warning_count": len(run.warnings),
        "artifact_count": len(run.artifacts),
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def _safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    run_dir = (root / thread_id).resolve()
    if root != run_dir and root not in run_dir.parents:
        raise ValueError("Invalid thread_id")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(thread_id)
    return run_dir


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
