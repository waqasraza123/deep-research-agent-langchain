"""Deterministic run artifact integrity reports for existing run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import REQUIRED_FILES, list_artifacts, now_iso_utc, safe_thread_id
from deep_research_agent.provenance.lineage import file_sha256

from .custody import read_run_custody_certificate
from .export_bundle import export_bundle_path, read_export_manifest
from .operator_audit import verify_operator_audit

IntegrityStatus = Literal["passed", "warning", "failed"]
IntegrityReadiness = Literal["valid", "warnings", "failed"]

INTEGRITY_REPORT_JSON = "integrity_report.json"
INTEGRITY_REPORT_MD = "integrity_report.md"
CONTROL_ARTIFACTS = {
    "operator_audit.jsonl",
    "operator_audit.md",
    "custody_certificate.json",
    "custody_certificate.md",
    INTEGRITY_REPORT_JSON,
    INTEGRITY_REPORT_MD,
}


class RunIntegrityRequest(BaseModel):
    requested_by: str = "operator"
    require_provenance_manifest: bool = False
    require_custody_certificate: bool = False
    require_export_bundle: bool = False
    include_artifact_hashes: bool = True
    max_artifacts: int = Field(default=1000, ge=1, le=10000)
    notes: str = ""


class IntegrityFinding(BaseModel):
    finding_id: str
    title: str
    status: IntegrityStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrityArtifactRecord(BaseModel):
    path: str
    size_bytes: int
    sha256: str


class IntegrityArtifactInventory(BaseModel):
    artifact_count: int = 0
    hashed_count: int = 0
    total_bytes: int = 0
    truncated: bool = False
    excluded_control_artifacts: list[str] = Field(default_factory=list)
    artifacts: list[IntegrityArtifactRecord] = Field(default_factory=list)


class RunIntegrityReport(BaseModel):
    report_version: str = "1.0"
    thread_id: str
    generated_at: str
    requested_by: str = "operator"
    readiness: IntegrityReadiness = "warnings"
    findings: list[IntegrityFinding] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifact_inventory: IntegrityArtifactInventory = Field(
        default_factory=IntegrityArtifactInventory
    )
    notes: str = ""


def build_run_integrity_report(
    *,
    runs_dir: Path,
    thread_id: str,
    request: RunIntegrityRequest | None = None,
) -> RunIntegrityReport:
    request = request or RunIntegrityRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    run_dir = _safe_run_dir(runs_dir, thread_id)
    artifact_paths = _artifact_path_set(runs_dir, thread_id)
    findings = [
        _required_artifacts_finding(artifact_paths),
        _provenance_manifest_finding(
            run_dir,
            artifact_paths,
            required=request.require_provenance_manifest,
        ),
        _custody_certificate_finding(
            runs_dir,
            thread_id,
            required=request.require_custody_certificate,
        ),
        _export_bundle_finding(
            runs_dir,
            thread_id,
            required=request.require_export_bundle,
        ),
        _operator_audit_finding(runs_dir, thread_id),
    ]
    artifact_inventory = (
        _artifact_inventory(
            runs_dir=runs_dir,
            thread_id=thread_id,
            max_artifacts=request.max_artifacts,
        )
        if request.include_artifact_hashes
        else IntegrityArtifactInventory(
            artifact_count=len(_stable_artifact_paths(artifact_paths)),
            excluded_control_artifacts=sorted(artifact_paths & CONTROL_ARTIFACTS),
        )
    )
    if artifact_inventory.truncated:
        findings.append(
            IntegrityFinding(
                finding_id="artifact_inventory_limit",
                title="Artifact Inventory Limit",
                status="warning",
                summary="Artifact inventory was truncated by max_artifacts.",
                required_action=(
                    "Increase max_artifacts and regenerate the integrity report for full hashing."
                ),
                metadata={"max_artifacts": request.max_artifacts},
            )
        )
    failures = [finding.summary for finding in findings if finding.status == "failed"]
    warnings = [finding.summary for finding in findings if finding.status == "warning"]
    readiness: IntegrityReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"
    report = RunIntegrityReport(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        findings=findings,
        failures=failures,
        warnings=warnings,
        artifact_inventory=artifact_inventory,
        notes=request.notes,
    )
    write_run_integrity_report(run_dir, report)
    return report


def read_run_integrity_report(runs_dir: Path, thread_id: str) -> RunIntegrityReport:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    path = run_dir / INTEGRITY_REPORT_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(RunIntegrityReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return RunIntegrityReport.parse_obj(data)


def write_run_integrity_report(run_dir: Path, report: RunIntegrityReport) -> list[str]:
    (run_dir / INTEGRITY_REPORT_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / INTEGRITY_REPORT_MD).write_text(
        render_run_integrity_report_markdown(report),
        encoding="utf-8",
    )
    return [INTEGRITY_REPORT_JSON, INTEGRITY_REPORT_MD]


def render_run_integrity_report_markdown(report: RunIntegrityReport) -> str:
    lines = [
        "# Run Integrity Report",
        "",
        f"- Thread ID: `{report.thread_id}`",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Failures: {len(report.failures)}",
        f"- Warnings: {len(report.warnings)}",
        "",
        "## Findings",
        "",
    ]
    for finding in report.findings:
        lines.extend(
            [
                f"### {finding.title}",
                "",
                f"- Status: `{finding.status}`",
                f"- Summary: {finding.summary or 'None'}",
                f"- Required action: {finding.required_action or 'None'}",
                "- Evidence: "
                + (", ".join(f"`{item}`" for item in finding.evidence_artifacts) or "none"),
                "",
            ]
        )
    if report.failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in report.failures)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report.warnings)
        lines.append("")
    lines.extend(
        [
            "## Artifact Inventory",
            "",
            f"- Artifact count: {report.artifact_inventory.artifact_count}",
            f"- Hashed count: {report.artifact_inventory.hashed_count}",
            f"- Total bytes: {report.artifact_inventory.total_bytes}",
            f"- Truncated: `{report.artifact_inventory.truncated}`",
        ]
    )
    if report.artifact_inventory.excluded_control_artifacts:
        excluded = ", ".join(
            f"`{item}`" for item in report.artifact_inventory.excluded_control_artifacts
        )
        lines.append(f"- Excluded control artifacts: {excluded}")
    lines.append("")
    if report.artifact_inventory.artifacts:
        lines.extend(["### Artifact Hashes", ""])
        for artifact in report.artifact_inventory.artifacts:
            lines.append(
                f"- `{artifact.path}` ({artifact.size_bytes} bytes) sha256=`{artifact.sha256}`"
            )
        lines.append("")
    if report.notes:
        lines.extend(["## Notes", "", report.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _required_artifacts_finding(artifact_paths: set[str]) -> IntegrityFinding:
    missing = [path for path in REQUIRED_FILES if path not in artifact_paths]
    if missing:
        return IntegrityFinding(
            finding_id="required_artifacts",
            title="Required Artifacts",
            status="failed",
            summary=f"Missing required artifacts: {', '.join(missing)}.",
            required_action="Regenerate or repair the run before relying on integrity results.",
            evidence_artifacts=[path for path in REQUIRED_FILES if path in artifact_paths],
            metadata={"missing": missing},
        )
    return IntegrityFinding(
        finding_id="required_artifacts",
        title="Required Artifacts",
        status="passed",
        summary="All required artifacts are present.",
        evidence_artifacts=list(REQUIRED_FILES),
    )


def _provenance_manifest_finding(
    run_dir: Path,
    artifact_paths: set[str],
    *,
    required: bool,
) -> IntegrityFinding:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.exists() or manifest_path.is_dir():
        return IntegrityFinding(
            finding_id="provenance_manifest",
            title="Provenance Manifest",
            status="failed" if required else "warning",
            summary="Provenance artifact manifest is missing.",
            required_action="Refresh provenance artifacts before integrity signoff.",
            metadata={"required": required},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return IntegrityFinding(
            finding_id="provenance_manifest",
            title="Provenance Manifest",
            status="failed",
            summary=f"Provenance artifact manifest is unreadable: {type(e).__name__}: {e}.",
            required_action="Regenerate provenance artifacts.",
            evidence_artifacts=["artifact_manifest.json"],
        )
    if not isinstance(manifest, dict):
        return IntegrityFinding(
            finding_id="provenance_manifest",
            title="Provenance Manifest",
            status="failed",
            summary="Provenance artifact manifest is not a JSON object.",
            required_action="Regenerate provenance artifacts.",
            evidence_artifacts=["artifact_manifest.json"],
        )
    entries = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("artifact_path"), str)
    ]
    missing: list[str] = []
    changed: dict[str, dict[str, str | None]] = {}
    checked = 0
    for item in entries:
        path = item["artifact_path"]
        expected_hash = item.get("content_hash")
        if path in CONTROL_ARTIFACTS or str(expected_hash) == "self-referential":
            continue
        if path not in artifact_paths:
            missing.append(path)
            continue
        checked += 1
        actual_hash = file_sha256(run_dir / path)
        if expected_hash and actual_hash != expected_hash:
            changed[path] = {"expected": expected_hash, "actual": actual_hash}
    if missing or changed:
        parts = []
        if missing:
            parts.append(f"{len(missing)} manifest artifact(s) are missing")
        if changed:
            parts.append(f"{len(changed)} manifest artifact hash(es) changed")
        return IntegrityFinding(
            finding_id="provenance_manifest",
            title="Provenance Manifest",
            status="failed",
            summary="; ".join(parts) + ".",
            required_action="Investigate artifact drift or refresh provenance after intentional changes.",
            evidence_artifacts=["artifact_manifest.json"],
            metadata={
                "checked": checked,
                "missing": missing,
                "changed_hashes": changed,
            },
        )
    return IntegrityFinding(
        finding_id="provenance_manifest",
        title="Provenance Manifest",
        status="passed",
        summary=f"Provenance manifest verified for {checked} stable artifact(s).",
        evidence_artifacts=["artifact_manifest.json"],
        metadata={"checked": checked, "manifest_entries": len(entries)},
    )


def _custody_certificate_finding(
    runs_dir: Path,
    thread_id: str,
    *,
    required: bool,
) -> IntegrityFinding:
    try:
        certificate = read_run_custody_certificate(runs_dir, thread_id)
    except FileNotFoundError:
        return IntegrityFinding(
            finding_id="custody_certificate",
            title="Custody Certificate",
            status="failed" if required else "warning",
            summary="Custody certificate is missing.",
            required_action="Generate a custody certificate if handoff readiness must be verified.",
            metadata={"required": required},
        )
    except Exception as e:
        return IntegrityFinding(
            finding_id="custody_certificate",
            title="Custody Certificate",
            status="failed",
            summary=f"Custody certificate is unreadable: {type(e).__name__}: {e}.",
            required_action="Regenerate the custody certificate.",
        )
    run_dir = _safe_run_dir(runs_dir, thread_id)
    missing: list[str] = []
    changed: dict[str, dict[str, str | None]] = {}
    for artifact in certificate.artifact_inventory.artifacts:
        if artifact.path in CONTROL_ARTIFACTS:
            continue
        path = run_dir / artifact.path
        if not path.exists() or path.is_dir():
            missing.append(artifact.path)
            continue
        actual_hash = file_sha256(path)
        if actual_hash != artifact.sha256:
            changed[artifact.path] = {"expected": artifact.sha256, "actual": actual_hash}
    if missing or changed:
        return IntegrityFinding(
            finding_id="custody_certificate",
            title="Custody Certificate",
            status="failed",
            summary="Custody artifact inventory no longer matches current files.",
            required_action="Investigate artifact drift and regenerate custody after review.",
            evidence_artifacts=["custody_certificate.json", "custody_certificate.md"],
            metadata={"missing": missing, "changed_hashes": changed},
        )
    status: IntegrityStatus = "passed"
    if certificate.readiness != "ready":
        status = "warning"
    return IntegrityFinding(
        finding_id="custody_certificate",
        title="Custody Certificate",
        status=status,
        summary=f"Custody certificate is `{certificate.readiness}` and inventory hashes match.",
        required_action="Resolve custody warnings or blockers before final handoff."
        if status == "warning"
        else "",
        evidence_artifacts=["custody_certificate.json", "custody_certificate.md"],
        metadata={
            "readiness": certificate.readiness,
            "hashed_count": certificate.artifact_inventory.hashed_count,
            "blockers": certificate.blockers,
            "warnings": certificate.warnings,
        },
    )


def _export_bundle_finding(
    runs_dir: Path,
    thread_id: str,
    *,
    required: bool,
) -> IntegrityFinding:
    try:
        manifest = read_export_manifest(runs_dir, thread_id)
        archive_path = export_bundle_path(runs_dir, thread_id)
    except FileNotFoundError:
        return IntegrityFinding(
            finding_id="export_bundle",
            title="Export Bundle",
            status="failed" if required else "warning",
            summary="Export bundle is missing.",
            required_action="Create an export bundle if portable handoff integrity is required.",
            metadata={"required": required},
        )
    archive_sha256 = file_sha256(archive_path)
    if manifest.archive_sha256 and archive_sha256 != manifest.archive_sha256:
        return IntegrityFinding(
            finding_id="export_bundle",
            title="Export Bundle",
            status="failed",
            summary="Export bundle archive hash does not match its manifest.",
            required_action="Regenerate the export bundle before use.",
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
    return IntegrityFinding(
        finding_id="export_bundle",
        title="Export Bundle",
        status="passed",
        summary="Export bundle exists and matches its manifest hash.",
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


def _operator_audit_finding(runs_dir: Path, thread_id: str) -> IntegrityFinding:
    global_verification = verify_operator_audit(runs_dir)
    run_verification = verify_operator_audit(runs_dir, thread_id)
    metadata = {
        "global": _model_to_plain(global_verification),
        "run": _model_to_plain(run_verification),
    }
    if not global_verification.valid or not run_verification.valid:
        return IntegrityFinding(
            finding_id="operator_audit",
            title="Operator Audit",
            status="failed",
            summary="Operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before relying on run custody.",
            evidence_artifacts=["operator_audit.jsonl", "operator_audit.md"],
            metadata=metadata,
        )
    if run_verification.event_count == 0:
        return IntegrityFinding(
            finding_id="operator_audit",
            title="Operator Audit",
            status="warning",
            summary="Per-run operator audit has no events.",
            required_action="Review whether this run needs operator action evidence.",
            evidence_artifacts=["operator_audit.jsonl", "operator_audit.md"],
            metadata=metadata,
        )
    return IntegrityFinding(
        finding_id="operator_audit",
        title="Operator Audit",
        status="passed",
        summary="Global and per-run operator audit chains verify.",
        evidence_artifacts=["operator_audit.jsonl", "operator_audit.md"],
        metadata=metadata,
    )


def _artifact_inventory(
    *,
    runs_dir: Path,
    thread_id: str,
    max_artifacts: int,
) -> IntegrityArtifactInventory:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    artifacts = [
        artifact
        for artifact in list_artifacts(runs_dir, thread_id)
        if artifact.path not in CONTROL_ARTIFACTS
    ]
    selected = artifacts[:max_artifacts]
    records = [
        IntegrityArtifactRecord(
            path=artifact.path,
            size_bytes=artifact.size_bytes,
            sha256=file_sha256(run_dir / artifact.path),
        )
        for artifact in selected
    ]
    return IntegrityArtifactInventory(
        artifact_count=len(artifacts),
        hashed_count=len(records),
        total_bytes=sum(artifact.size_bytes for artifact in artifacts),
        truncated=len(artifacts) > len(selected),
        excluded_control_artifacts=sorted(
            artifact.path
            for artifact in list_artifacts(runs_dir, thread_id)
            if artifact.path in CONTROL_ARTIFACTS
        ),
        artifacts=records,
    )


def _artifact_path_set(runs_dir: Path, thread_id: str) -> set[str]:
    return {artifact.path for artifact in list_artifacts(runs_dir, thread_id)}


def _stable_artifact_paths(artifact_paths: set[str]) -> set[str]:
    return {path for path in artifact_paths if path not in CONTROL_ARTIFACTS}


def _safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    run_dir = (root / thread_id).resolve()
    if root != run_dir and root not in run_dir.parents:
        raise ValueError("Invalid thread_id")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(thread_id)
    return run_dir


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
