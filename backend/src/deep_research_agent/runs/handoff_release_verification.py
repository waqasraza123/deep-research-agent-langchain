"""Deterministic verification reports for handoff release manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .export_bundle import export_bundle_path
from .handoff_release import (
    HANDOFF_RELEASE_JSON,
    HANDOFF_RELEASE_MD,
    HANDOFF_RELEASES_DIR,
    HandoffReleaseManifest,
    read_handoff_release_manifest,
)
from .operator_audit import verify_operator_audit

VerificationFindingStatus = Literal["passed", "warning", "failed"]
ReleaseVerificationReadiness = Literal["valid", "warnings", "failed"]

HANDOFF_RELEASE_VERIFICATION_JSON = "handoff_release_verification.json"
HANDOFF_RELEASE_VERIFICATION_MD = "handoff_release_verification.md"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class HandoffReleaseVerificationRequest(BaseModel):
    requested_by: str = "operator"
    require_registry_hash_match: bool = True
    require_export_hashes: bool = True
    require_release_artifacts: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleaseVerificationFinding(BaseModel):
    finding_id: str
    title: str
    status: VerificationFindingStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleaseRunVerification(BaseModel):
    thread_id: str
    export_archive_status: VerificationFindingStatus = "warning"
    expected_export_sha256: str | None = None
    actual_export_sha256: str | None = None
    missing_artifacts: list[str] = Field(default_factory=list)
    findings: list[HandoffReleaseVerificationFinding] = Field(default_factory=list)


class HandoffReleaseVerificationReport(BaseModel):
    report_version: str = "1.0"
    release_id: str
    generated_at: str
    requested_by: str = "operator"
    readiness: ReleaseVerificationReadiness = "warnings"
    release_readiness: str
    release_generated_at: str
    release_sha256: str | None = None
    findings: list[HandoffReleaseVerificationFinding] = Field(default_factory=list)
    run_verifications: list[HandoffReleaseRunVerification] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_verification_report(
    *,
    runs_dir: Path,
    release_id: str,
    request: HandoffReleaseVerificationRequest | None = None,
    release: HandoffReleaseManifest | None = None,
) -> HandoffReleaseVerificationReport:
    request = request or HandoffReleaseVerificationRequest()
    release = release or read_handoff_release_manifest(runs_dir, release_id)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    release_dir = _release_dir(runs_dir, release.release_id)
    findings = [
        _release_artifacts_finding(
            release_dir,
            release.release_id,
            required=request.require_release_artifacts,
        ),
        _registry_hash_finding(
            runs_dir,
            release,
            required=request.require_registry_hash_match,
        ),
        _operator_audit_finding(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]
    run_verifications = [
        _run_verification(
            runs_dir,
            release_run,
            require_export_hash=request.require_export_hashes,
        )
        for release_run in release.runs
    ]
    for run_verification in run_verifications:
        findings.extend(run_verification.findings)

    failures = [finding.summary for finding in findings if finding.status == "failed"]
    warnings = [finding.summary for finding in findings if finding.status == "warning"]
    readiness: ReleaseVerificationReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"

    report = HandoffReleaseVerificationReport(
        release_id=release.release_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        release_readiness=release.readiness,
        release_generated_at=release.generated_at,
        release_sha256=_file_sha256_or_none(release_dir / HANDOFF_RELEASE_JSON),
        findings=findings,
        run_verifications=run_verifications,
        failures=failures,
        warnings=warnings,
        required_controls={
            "registry_hash_match": request.require_registry_hash_match,
            "export_hashes": request.require_export_hashes,
            "release_artifacts": request.require_release_artifacts,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            _release_rel_path(release.release_id, HANDOFF_RELEASE_VERIFICATION_JSON),
            _release_rel_path(release.release_id, HANDOFF_RELEASE_VERIFICATION_MD),
        ],
        notes=request.notes,
    )
    write_handoff_release_verification_report(runs_dir, report)
    return report


def read_handoff_release_verification_report(
    runs_dir: Path,
    release_id: str,
) -> HandoffReleaseVerificationReport:
    path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_VERIFICATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseVerificationReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseVerificationReport.parse_obj(data)


def write_handoff_release_verification_report(
    runs_dir: Path,
    report: HandoffReleaseVerificationReport,
) -> list[str]:
    release_dir = _release_dir(runs_dir, report.release_id)
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / HANDOFF_RELEASE_VERIFICATION_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / HANDOFF_RELEASE_VERIFICATION_MD).write_text(
        render_handoff_release_verification_markdown(report),
        encoding="utf-8",
    )
    return [
        _release_rel_path(report.release_id, HANDOFF_RELEASE_VERIFICATION_JSON),
        _release_rel_path(report.release_id, HANDOFF_RELEASE_VERIFICATION_MD),
    ]


def render_handoff_release_verification_markdown(
    report: HandoffReleaseVerificationReport,
) -> str:
    lines = [
        "# Handoff Release Verification",
        "",
        f"- Release ID: `{report.release_id}`",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Release readiness: `{report.release_readiness}`",
        f"- Release generated at: `{report.release_generated_at}`",
        f"- Release SHA-256: `{report.release_sha256 or 'unavailable'}`",
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
    if report.run_verifications:
        lines.extend(["## Run Verification", ""])
        for run in report.run_verifications:
            lines.extend(
                [
                    f"### {run.thread_id}",
                    "",
                    f"- Export archive status: `{run.export_archive_status}`",
                    f"- Expected export SHA-256: `{run.expected_export_sha256 or 'unavailable'}`",
                    f"- Actual export SHA-256: `{run.actual_export_sha256 or 'unavailable'}`",
                    "- Missing artifacts: "
                    + (", ".join(f"`{item}`" for item in run.missing_artifacts) or "none"),
                    "",
                ]
            )
    if report.failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in report.failures)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    if report.notes:
        lines.extend(["## Notes", "", report.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _release_artifacts_finding(
    release_dir: Path,
    release_id: str,
    *,
    required: bool,
) -> HandoffReleaseVerificationFinding:
    expected = [HANDOFF_RELEASE_JSON, HANDOFF_RELEASE_MD]
    missing = [name for name in expected if not (release_dir / name).exists()]
    if missing:
        return HandoffReleaseVerificationFinding(
            finding_id="release_artifacts",
            title="Release Artifacts",
            status="failed" if required else "warning",
            summary="Release artifacts are missing: " + ", ".join(missing),
            required_action="Regenerate or restore the release manifest artifacts.",
            evidence_artifacts=[_release_rel_path(release_id, name) for name in expected],
            metadata={"missing": missing, "required": required},
        )
    return HandoffReleaseVerificationFinding(
        finding_id="release_artifacts",
        title="Release Artifacts",
        status="passed",
        summary="Release JSON and Markdown artifacts exist.",
        evidence_artifacts=[_release_rel_path(release_id, name) for name in expected],
        metadata={"required": required},
    )


def _registry_hash_finding(
    runs_dir: Path,
    release: HandoffReleaseManifest,
    *,
    required: bool,
) -> HandoffReleaseVerificationFinding:
    registry_path = runs_dir / "_handoff" / "handoff_registry.json"
    actual_hash = _file_sha256_or_none(registry_path)
    expected_hash = release.registry_sha256
    metadata = {
        "expected_registry_sha256": expected_hash,
        "actual_registry_sha256": actual_hash,
        "required": required,
    }
    if expected_hash is None:
        return HandoffReleaseVerificationFinding(
            finding_id="registry_hash",
            title="Registry Snapshot Hash",
            status="failed" if required else "warning",
            summary="Release manifest does not record a registry SHA-256.",
            required_action="Regenerate the release from a handoff registry with a recorded hash.",
            evidence_artifacts=["_handoff/handoff_registry.json"],
            metadata=metadata,
        )
    if actual_hash is None:
        return HandoffReleaseVerificationFinding(
            finding_id="registry_hash",
            title="Registry Snapshot Hash",
            status="failed" if required else "warning",
            summary="Current handoff registry artifact is missing.",
            required_action="Restore or regenerate the handoff registry before verification.",
            evidence_artifacts=["_handoff/handoff_registry.json"],
            metadata=metadata,
        )
    if expected_hash != actual_hash:
        return HandoffReleaseVerificationFinding(
            finding_id="registry_hash",
            title="Registry Snapshot Hash",
            status="failed" if required else "warning",
            summary="Current handoff registry hash differs from the release snapshot hash.",
            required_action="Confirm whether the release was based on an older registry snapshot.",
            evidence_artifacts=["_handoff/handoff_registry.json"],
            metadata=metadata,
        )
    return HandoffReleaseVerificationFinding(
        finding_id="registry_hash",
        title="Registry Snapshot Hash",
        status="passed",
        summary="Current handoff registry hash matches the release snapshot hash.",
        evidence_artifacts=["_handoff/handoff_registry.json"],
        metadata=metadata,
    )


def _operator_audit_finding(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleaseVerificationFinding:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleaseVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleaseVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before release reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleaseVerificationFinding(
        finding_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _run_verification(
    runs_dir: Path,
    release_run: Any,
    *,
    require_export_hash: bool,
) -> HandoffReleaseRunVerification:
    findings: list[HandoffReleaseVerificationFinding] = []
    missing_artifacts: list[str] = []
    export_status: VerificationFindingStatus = "passed"
    expected_export_sha256 = release_run.export_archive_sha256
    actual_export_sha256: str | None = None
    try:
        export_path = export_bundle_path(runs_dir, release_run.thread_id)
        actual_export_sha256 = file_sha256(export_path)
    except FileNotFoundError:
        missing_artifacts.append("exports/run_export.zip")
        export_status = "failed" if require_export_hash else "warning"
        findings.append(
            HandoffReleaseVerificationFinding(
                finding_id=f"{release_run.thread_id}:export_archive",
                title="Run Export Archive",
                status=export_status,
                summary=f"{release_run.thread_id}: export archive is missing.",
                required_action="Regenerate the run export bundle before release reliance.",
                evidence_artifacts=[f"{release_run.thread_id}/exports/run_export.zip"],
                metadata={"required": require_export_hash},
            )
        )
    else:
        if expected_export_sha256 and expected_export_sha256 != actual_export_sha256:
            export_status = "failed" if require_export_hash else "warning"
            findings.append(
                HandoffReleaseVerificationFinding(
                    finding_id=f"{release_run.thread_id}:export_archive",
                    title="Run Export Archive",
                    status=export_status,
                    summary=f"{release_run.thread_id}: export archive hash changed.",
                    required_action="Regenerate registry and release after export changes.",
                    evidence_artifacts=[f"{release_run.thread_id}/exports/run_export.zip"],
                    metadata={
                        "expected_export_sha256": expected_export_sha256,
                        "actual_export_sha256": actual_export_sha256,
                        "required": require_export_hash,
                    },
                )
            )
        else:
            findings.append(
                HandoffReleaseVerificationFinding(
                    finding_id=f"{release_run.thread_id}:export_archive",
                    title="Run Export Archive",
                    status="passed",
                    summary=f"{release_run.thread_id}: export archive hash matches release record.",
                    evidence_artifacts=[f"{release_run.thread_id}/exports/run_export.zip"],
                    metadata={
                        "expected_export_sha256": expected_export_sha256,
                        "actual_export_sha256": actual_export_sha256,
                        "required": require_export_hash,
                    },
                )
            )
    for rel_path in release_run.artifacts:
        path = runs_dir / release_run.thread_id / rel_path
        if not path.exists():
            missing_artifacts.append(rel_path)
    if missing_artifacts:
        findings.append(
            HandoffReleaseVerificationFinding(
                finding_id=f"{release_run.thread_id}:run_artifacts",
                title="Run Release Artifacts",
                status="warning",
                summary=f"{release_run.thread_id}: release references missing run artifacts.",
                required_action="Inspect run artifact drift before relying on the release.",
                evidence_artifacts=[
                    f"{release_run.thread_id}/{item}" for item in missing_artifacts
                ],
                metadata={"missing_artifacts": sorted(set(missing_artifacts))},
            )
        )
    return HandoffReleaseRunVerification(
        thread_id=release_run.thread_id,
        export_archive_status=export_status,
        expected_export_sha256=expected_export_sha256,
        actual_export_sha256=actual_export_sha256,
        missing_artifacts=sorted(set(missing_artifacts)),
        findings=findings,
    )


def _file_sha256_or_none(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    return file_sha256(path)


def _release_dir(runs_dir: Path, release_id: str) -> Path:
    release_id = _safe_release_id(release_id)
    root = runs_dir.resolve()
    release_root = (root / HANDOFF_RELEASES_DIR).resolve()
    release_dir = (release_root / release_id).resolve()
    if release_root != release_dir and release_root not in release_dir.parents:
        raise ValueError("Invalid release_id")
    return release_dir


def _release_rel_path(release_id: str, filename: str) -> str:
    return f"{HANDOFF_RELEASES_DIR}/{release_id}/{filename}"


def _safe_release_id(release_id: str) -> str:
    release_id = release_id.strip()
    if not RELEASE_ID_PATTERN.match(release_id):
        raise ValueError("Invalid release_id")
    return release_id


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
