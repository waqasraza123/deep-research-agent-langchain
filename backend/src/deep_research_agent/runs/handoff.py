"""Final deterministic handoff manifests for reviewed run packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc, safe_thread_id
from deep_research_agent.provenance.lineage import file_sha256

from .contracts import ResearchRun
from .custody import read_run_custody_certificate
from .disclosure import read_run_disclosure_report
from .export_bundle import export_bundle_path, read_export_manifest
from .integrity import read_run_integrity_report
from .operator_audit import verify_operator_audit
from .retention import read_retention_policy

HandoffGateStatus = Literal["passed", "warning", "blocked"]
HandoffReadiness = Literal["ready_for_handoff", "needs_attention", "blocked"]

HANDOFF_MANIFEST_JSON = "handoff_manifest.json"
HANDOFF_MANIFEST_MD = "handoff_manifest.md"


class RunHandoffRequest(BaseModel):
    requested_by: str = "operator"
    recipient: str = ""
    purpose: str = "external_handoff"
    require_review_approval: bool = False
    require_export_bundle: bool = True
    require_retention_policy: bool = True
    require_custody_ready: bool = True
    require_integrity_valid: bool = True
    require_disclosure_clear: bool = True
    notes: str = ""


class HandoffGate(BaseModel):
    gate_id: str
    title: str
    status: HandoffGateStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunHandoffManifest(BaseModel):
    manifest_version: str = "1.0"
    thread_id: str
    generated_at: str
    requested_by: str = "operator"
    recipient: str = ""
    purpose: str = "external_handoff"
    readiness: HandoffReadiness = "needs_attention"
    gates: list[HandoffGate] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    handoff_artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_run_handoff_manifest(
    *,
    runs_dir: Path,
    thread_id: str,
    run: ResearchRun | None = None,
    request: RunHandoffRequest | None = None,
) -> RunHandoffManifest:
    request = request or RunHandoffRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    run_dir = _safe_run_dir(runs_dir, thread_id)
    gates = [
        _run_state_gate(run),
        _review_gate(run, required=request.require_review_approval),
        _retention_gate(runs_dir, thread_id, required=request.require_retention_policy),
        _export_gate(runs_dir, thread_id, required=request.require_export_bundle),
        _custody_gate(runs_dir, thread_id, require_ready=request.require_custody_ready),
        _integrity_gate(runs_dir, thread_id, require_valid=request.require_integrity_valid),
        _disclosure_gate(runs_dir, thread_id, require_clear=request.require_disclosure_clear),
        _operator_audit_gate(runs_dir, thread_id),
    ]
    blockers = [gate.summary for gate in gates if gate.status == "blocked" and gate.summary]
    warnings = [gate.summary for gate in gates if gate.status == "warning" and gate.summary]
    readiness: HandoffReadiness = "ready_for_handoff"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "needs_attention"
    manifest = RunHandoffManifest(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        recipient=request.recipient,
        purpose=request.purpose,
        readiness=readiness,
        gates=gates,
        blockers=blockers,
        warnings=warnings,
        required_controls={
            "review_approval": request.require_review_approval,
            "export_bundle": request.require_export_bundle,
            "retention_policy": request.require_retention_policy,
            "custody_ready": request.require_custody_ready,
            "integrity_valid": request.require_integrity_valid,
            "disclosure_clear": request.require_disclosure_clear,
        },
        handoff_artifacts=_handoff_artifacts(run_dir),
        notes=request.notes,
    )
    write_run_handoff_manifest(run_dir, manifest)
    return manifest


def read_run_handoff_manifest(runs_dir: Path, thread_id: str) -> RunHandoffManifest:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    path = run_dir / HANDOFF_MANIFEST_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(RunHandoffManifest, "model_validate", None)
    if callable(validate):
        return validate(data)
    return RunHandoffManifest.parse_obj(data)


def write_run_handoff_manifest(run_dir: Path, manifest: RunHandoffManifest) -> list[str]:
    (run_dir / HANDOFF_MANIFEST_JSON).write_text(
        json.dumps(_model_to_plain(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / HANDOFF_MANIFEST_MD).write_text(
        render_run_handoff_manifest_markdown(manifest),
        encoding="utf-8",
    )
    return [HANDOFF_MANIFEST_JSON, HANDOFF_MANIFEST_MD]


def render_run_handoff_manifest_markdown(manifest: RunHandoffManifest) -> str:
    lines = [
        "# Run Handoff Manifest",
        "",
        f"- Thread ID: `{manifest.thread_id}`",
        f"- Generated at: `{manifest.generated_at}`",
        f"- Requested by: `{manifest.requested_by}`",
        f"- Recipient: {manifest.recipient or 'unspecified'}",
        f"- Purpose: `{manifest.purpose}`",
        f"- Readiness: `{manifest.readiness}`",
        f"- Blockers: {len(manifest.blockers)}",
        f"- Warnings: {len(manifest.warnings)}",
        "",
        "## Gates",
        "",
    ]
    for gate in manifest.gates:
        lines.extend(
            [
                f"### {gate.title}",
                "",
                f"- Status: `{gate.status}`",
                f"- Summary: {gate.summary or 'None'}",
                f"- Required action: {gate.required_action or 'None'}",
                "- Evidence: "
                + (", ".join(f"`{item}`" for item in gate.evidence_artifacts) or "none"),
                "",
            ]
        )
    if manifest.blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {item}" for item in manifest.blockers)
        lines.append("")
    if manifest.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in manifest.warnings)
        lines.append("")
    lines.extend(["## Handoff Artifacts", ""])
    if manifest.handoff_artifacts:
        lines.extend(f"- `{item}`" for item in manifest.handoff_artifacts)
    else:
        lines.append("- None")
    lines.append("")
    if manifest.notes:
        lines.extend(["## Notes", "", manifest.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _run_state_gate(run: ResearchRun | None) -> HandoffGate:
    if run is None:
        return HandoffGate(
            gate_id="run_state",
            title="Run State",
            status="warning",
            summary="Run metadata is unavailable; handoff relies on artifacts only.",
            required_action="Restore .run.json if lifecycle state must be included.",
        )
    status = _enum_value(run.status)
    if status in {"failed", "cancelled"}:
        return HandoffGate(
            gate_id="run_state",
            title="Run State",
            status="blocked",
            summary=f"Run is `{status}`.",
            required_action="Resolve or explicitly document failed/cancelled state before handoff.",
            metadata={"status": status},
        )
    if status == "waiting_for_review":
        return HandoffGate(
            gate_id="run_state",
            title="Run State",
            status="warning",
            summary="Run is waiting for review.",
            required_action="Approve, reject, or request changes before final handoff.",
            metadata={"status": status},
        )
    return HandoffGate(
        gate_id="run_state",
        title="Run State",
        status="passed",
        summary=f"Run state is `{status}`.",
        metadata={"status": status},
    )


def _review_gate(run: ResearchRun | None, *, required: bool) -> HandoffGate:
    review_status = "unknown"
    reviewer = None
    if run is not None:
        review_status = _enum_value(run.review.status)
        reviewer = run.review.reviewer
    metadata = {"review_status": review_status, "reviewer": reviewer, "required": required}
    if review_status == "approved":
        return HandoffGate(
            gate_id="review",
            title="Review Approval",
            status="passed",
            summary="Review status is approved.",
            evidence_artifacts=["review_dossier.json", "review_dossier.md"],
            metadata=metadata,
        )
    if review_status == "rejected":
        return HandoffGate(
            gate_id="review",
            title="Review Approval",
            status="blocked",
            summary="Review status is rejected.",
            required_action="Do not hand off a rejected run.",
            evidence_artifacts=["review_dossier.json", "review_dossier.md"],
            metadata=metadata,
        )
    if review_status == "changes_requested":
        return HandoffGate(
            gate_id="review",
            title="Review Approval",
            status="blocked",
            summary="Review status is changes requested.",
            required_action="Resolve requested changes before handoff.",
            evidence_artifacts=["review_dossier.json", "review_dossier.md"],
            metadata=metadata,
        )
    if review_status == "not_required" and not required:
        return HandoffGate(
            gate_id="review",
            title="Review Approval",
            status="passed",
            summary="Review approval is not required for this run.",
            evidence_artifacts=["review_dossier.json", "review_dossier.md"],
            metadata=metadata,
        )
    return HandoffGate(
        gate_id="review",
        title="Review Approval",
        status="blocked" if required else "warning",
        summary=f"Review approval is not present; current status is `{review_status}`.",
        required_action="Approve the run or document why handoff proceeds without approval.",
        evidence_artifacts=["review_dossier.json", "review_dossier.md"],
        metadata=metadata,
    )


def _retention_gate(runs_dir: Path, thread_id: str, *, required: bool) -> HandoffGate:
    try:
        policy = read_retention_policy(runs_dir, thread_id)
    except FileNotFoundError:
        return HandoffGate(
            gate_id="retention",
            title="Retention Policy",
            status="blocked" if required else "warning",
            summary="Retention policy is missing.",
            required_action="Set retention policy before handoff if the run must remain protected.",
            metadata={"required": required},
        )
    return HandoffGate(
        gate_id="retention",
        title="Retention Policy",
        status="passed",
        summary=f"Retention policy `{policy.retention_class}` is present.",
        evidence_artifacts=["retention_policy.json", "retention_policy.md"],
        metadata={
            "retention_class": policy.retention_class,
            "retain_until": policy.retain_until,
            "delete_after": policy.delete_after,
            "legal_hold": policy.legal_hold,
            "active_hold_ids": [hold.hold_id for hold in policy.active_holds],
        },
    )


def _export_gate(runs_dir: Path, thread_id: str, *, required: bool) -> HandoffGate:
    try:
        manifest = read_export_manifest(runs_dir, thread_id)
        archive_path = export_bundle_path(runs_dir, thread_id)
    except FileNotFoundError:
        return HandoffGate(
            gate_id="export",
            title="Export Bundle",
            status="blocked" if required else "warning",
            summary="Export bundle is missing.",
            required_action="Create an export bundle before handoff.",
            metadata={"required": required},
        )
    archive_sha256 = file_sha256(archive_path)
    if manifest.archive_sha256 and archive_sha256 != manifest.archive_sha256:
        return HandoffGate(
            gate_id="export",
            title="Export Bundle",
            status="blocked",
            summary="Export archive hash does not match export manifest.",
            required_action="Regenerate the export bundle.",
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
    return HandoffGate(
        gate_id="export",
        title="Export Bundle",
        status="passed",
        summary="Export bundle exists and matches manifest hash.",
        evidence_artifacts=[
            "exports/run_export.zip",
            "exports/export_manifest.json",
            "exports/export_manifest.md",
        ],
        metadata={
            "profile": manifest.profile,
            "redact": manifest.redact,
            "include_raw_sources": manifest.include_raw_sources,
            "archive_sha256": archive_sha256,
            "exported_count": manifest.exported_count,
            "skipped_count": manifest.skipped_count,
        },
    )


def _custody_gate(runs_dir: Path, thread_id: str, *, require_ready: bool) -> HandoffGate:
    try:
        certificate = read_run_custody_certificate(runs_dir, thread_id)
    except FileNotFoundError:
        return HandoffGate(
            gate_id="custody",
            title="Custody Certificate",
            status="blocked" if require_ready else "warning",
            summary="Custody certificate is missing.",
            required_action="Generate custody certificate before handoff.",
            metadata={"required": require_ready},
        )
    status: HandoffGateStatus = "passed"
    if certificate.readiness == "blocked" or (require_ready and certificate.readiness != "ready"):
        status = "blocked"
    elif certificate.readiness != "ready":
        status = "warning"
    return HandoffGate(
        gate_id="custody",
        title="Custody Certificate",
        status=status,
        summary=f"Custody readiness is `{certificate.readiness}`.",
        required_action="Resolve custody blockers or warnings before handoff."
        if status != "passed"
        else "",
        evidence_artifacts=["custody_certificate.json", "custody_certificate.md"],
        metadata={
            "readiness": certificate.readiness,
            "blocker_count": len(certificate.blockers),
            "warning_count": len(certificate.warnings),
        },
    )


def _integrity_gate(runs_dir: Path, thread_id: str, *, require_valid: bool) -> HandoffGate:
    try:
        report = read_run_integrity_report(runs_dir, thread_id)
    except FileNotFoundError:
        return HandoffGate(
            gate_id="integrity",
            title="Integrity Report",
            status="blocked" if require_valid else "warning",
            summary="Integrity report is missing.",
            required_action="Generate integrity report before handoff.",
            metadata={"required": require_valid},
        )
    status: HandoffGateStatus = "passed"
    if report.readiness == "failed" or (require_valid and report.readiness != "valid"):
        status = "blocked"
    elif report.readiness != "valid":
        status = "warning"
    return HandoffGate(
        gate_id="integrity",
        title="Integrity Report",
        status=status,
        summary=f"Integrity readiness is `{report.readiness}`.",
        required_action="Resolve integrity failures or warnings before handoff."
        if status != "passed"
        else "",
        evidence_artifacts=["integrity_report.json", "integrity_report.md"],
        metadata={
            "readiness": report.readiness,
            "failure_count": len(report.failures),
            "warning_count": len(report.warnings),
        },
    )


def _disclosure_gate(runs_dir: Path, thread_id: str, *, require_clear: bool) -> HandoffGate:
    try:
        report = read_run_disclosure_report(runs_dir, thread_id)
    except FileNotFoundError:
        return HandoffGate(
            gate_id="disclosure",
            title="Disclosure Report",
            status="blocked" if require_clear else "warning",
            summary="Disclosure report is missing.",
            required_action="Generate disclosure report before external handoff.",
            metadata={"required": require_clear},
        )
    status: HandoffGateStatus = "passed"
    if report.readiness == "blocked" or (require_clear and report.readiness != "clear"):
        status = "blocked"
    elif report.readiness != "clear":
        status = "warning"
    return HandoffGate(
        gate_id="disclosure",
        title="Disclosure Report",
        status=status,
        summary=f"Disclosure readiness is `{report.readiness}` with risk `{report.risk_level}`.",
        required_action="Resolve disclosure findings before external handoff."
        if status != "passed"
        else "",
        evidence_artifacts=["disclosure_report.json", "disclosure_report.md"],
        metadata={
            "readiness": report.readiness,
            "risk_level": report.risk_level,
            "finding_count": len(report.findings),
            "high_or_critical_count": report.high_or_critical_count,
        },
    )


def _operator_audit_gate(runs_dir: Path, thread_id: str) -> HandoffGate:
    global_verification = verify_operator_audit(runs_dir)
    run_verification = verify_operator_audit(runs_dir, thread_id)
    metadata = {
        "global": _model_to_plain(global_verification),
        "run": _model_to_plain(run_verification),
    }
    if not global_verification.valid or not run_verification.valid:
        return HandoffGate(
            gate_id="operator_audit",
            title="Operator Audit",
            status="blocked",
            summary="Operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before handoff.",
            evidence_artifacts=["operator_audit.jsonl", "operator_audit.md"],
            metadata=metadata,
        )
    if run_verification.event_count == 0:
        return HandoffGate(
            gate_id="operator_audit",
            title="Operator Audit",
            status="warning",
            summary="Per-run operator audit has no events.",
            required_action="Record required operator actions before final handoff if applicable.",
            evidence_artifacts=["operator_audit.jsonl", "operator_audit.md"],
            metadata=metadata,
        )
    return HandoffGate(
        gate_id="operator_audit",
        title="Operator Audit",
        status="passed",
        summary="Global and per-run audit chains verify.",
        evidence_artifacts=["operator_audit.jsonl", "operator_audit.md"],
        metadata=metadata,
    )


def _handoff_artifacts(run_dir: Path) -> list[str]:
    expected = [
        "review_dossier.md",
        "retention_policy.md",
        "custody_certificate.md",
        "integrity_report.md",
        "disclosure_report.md",
        "exports/export_manifest.md",
        "exports/run_export.zip",
    ]
    return [path for path in expected if (run_dir / path).exists()]


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
