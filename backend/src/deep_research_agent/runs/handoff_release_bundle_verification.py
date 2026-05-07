"""Deterministic verification reports for handoff release bundle archives."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release import HANDOFF_RELEASES_DIR
from .handoff_release_bundle import (
    HANDOFF_RELEASE_BUNDLE_JSON,
    HANDOFF_RELEASE_BUNDLE_MD,
    HANDOFF_RELEASE_BUNDLE_NAME,
    HandoffReleaseBundleManifest,
    handoff_release_bundle_path,
    read_handoff_release_bundle_manifest,
)
from .operator_audit import verify_operator_audit

BundleVerificationStatus = Literal["passed", "warning", "failed"]
BundleVerificationReadiness = Literal["valid", "warnings", "failed"]

HANDOFF_RELEASE_BUNDLE_VERIFICATION_JSON = "handoff_release_bundle_verification.json"
HANDOFF_RELEASE_BUNDLE_VERIFICATION_MD = "handoff_release_bundle_verification.md"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class HandoffReleaseBundleVerificationRequest(BaseModel):
    requested_by: str = "operator"
    require_archive_hash_match: bool = True
    require_manifest_entries: bool = True
    require_path_safety: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleaseBundleVerificationFinding(BaseModel):
    finding_id: str
    title: str
    status: BundleVerificationStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleaseBundleEntryVerification(BaseModel):
    archive_path: str
    entry_type: str
    expected_sha256: str
    actual_sha256: str | None = None
    expected_size_bytes: int
    actual_size_bytes: int | None = None
    status: BundleVerificationStatus = "warning"
    summary: str = ""


class HandoffReleaseBundleVerificationReport(BaseModel):
    report_version: str = "1.0"
    release_id: str
    generated_at: str
    requested_by: str = "operator"
    readiness: BundleVerificationReadiness = "warnings"
    bundle_readiness: str
    bundle_generated_at: str
    bundle_archive_sha256: str | None = None
    expected_bundle_archive_sha256: str | None = None
    findings: list[HandoffReleaseBundleVerificationFinding] = Field(default_factory=list)
    entry_verifications: list[HandoffReleaseBundleEntryVerification] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_bundle_verification_report(
    *,
    runs_dir: Path,
    release_id: str,
    request: HandoffReleaseBundleVerificationRequest | None = None,
    bundle: HandoffReleaseBundleManifest | None = None,
) -> HandoffReleaseBundleVerificationReport:
    request = request or HandoffReleaseBundleVerificationRequest()
    bundle = bundle or read_handoff_release_bundle_manifest(runs_dir, release_id)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    bundle_path = handoff_release_bundle_path(runs_dir, bundle.release_id)
    release_dir = _release_dir(runs_dir, bundle.release_id)

    findings = [
        _sidecar_artifacts_finding(release_dir, bundle.release_id),
        _archive_hash_finding(
            bundle_path,
            bundle,
            required=request.require_archive_hash_match,
        ),
        _operator_audit_finding(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]
    zip_findings, entry_verifications = _zip_inventory_findings(
        bundle_path,
        bundle,
        require_manifest_entries=request.require_manifest_entries,
        require_path_safety=request.require_path_safety,
    )
    findings.extend(zip_findings)

    failures = [finding.summary for finding in findings if finding.status == "failed"]
    warnings = [finding.summary for finding in findings if finding.status == "warning"]
    readiness: BundleVerificationReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"

    report = HandoffReleaseBundleVerificationReport(
        release_id=bundle.release_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        bundle_readiness=bundle.readiness,
        bundle_generated_at=bundle.generated_at,
        bundle_archive_sha256=_file_sha256_or_none(bundle_path),
        expected_bundle_archive_sha256=bundle.archive_sha256 or None,
        findings=findings,
        entry_verifications=entry_verifications,
        failures=failures,
        warnings=warnings,
        required_controls={
            "archive_hash_match": request.require_archive_hash_match,
            "manifest_entries": request.require_manifest_entries,
            "path_safety": request.require_path_safety,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            _release_rel_path(bundle.release_id, HANDOFF_RELEASE_BUNDLE_VERIFICATION_JSON),
            _release_rel_path(bundle.release_id, HANDOFF_RELEASE_BUNDLE_VERIFICATION_MD),
        ],
        notes=request.notes,
    )
    write_handoff_release_bundle_verification_report(runs_dir, report)
    return report


def read_handoff_release_bundle_verification_report(
    runs_dir: Path,
    release_id: str,
) -> HandoffReleaseBundleVerificationReport:
    path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_BUNDLE_VERIFICATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseBundleVerificationReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseBundleVerificationReport.parse_obj(data)


def write_handoff_release_bundle_verification_report(
    runs_dir: Path,
    report: HandoffReleaseBundleVerificationReport,
) -> list[str]:
    release_dir = _release_dir(runs_dir, report.release_id)
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / HANDOFF_RELEASE_BUNDLE_VERIFICATION_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / HANDOFF_RELEASE_BUNDLE_VERIFICATION_MD).write_text(
        render_handoff_release_bundle_verification_markdown(report),
        encoding="utf-8",
    )
    return [
        _release_rel_path(report.release_id, HANDOFF_RELEASE_BUNDLE_VERIFICATION_JSON),
        _release_rel_path(report.release_id, HANDOFF_RELEASE_BUNDLE_VERIFICATION_MD),
    ]


def render_handoff_release_bundle_verification_markdown(
    report: HandoffReleaseBundleVerificationReport,
) -> str:
    lines = [
        "# Handoff Release Bundle Verification",
        "",
        f"- Release ID: `{report.release_id}`",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Bundle readiness: `{report.bundle_readiness}`",
        f"- Bundle generated at: `{report.bundle_generated_at}`",
        f"- Expected bundle SHA-256: `{report.expected_bundle_archive_sha256 or 'unavailable'}`",
        f"- Actual bundle SHA-256: `{report.bundle_archive_sha256 or 'unavailable'}`",
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
    if report.entry_verifications:
        lines.extend(["## Entry Verification", ""])
        for entry in report.entry_verifications:
            lines.extend(
                [
                    f"- `{entry.archive_path}`: `{entry.status}` "
                    f"({entry.actual_size_bytes or 0}/{entry.expected_size_bytes} bytes, "
                    f"sha256=`{entry.actual_sha256 or 'missing'}`)",
                ]
            )
        lines.append("")
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


def _sidecar_artifacts_finding(
    release_dir: Path,
    release_id: str,
) -> HandoffReleaseBundleVerificationFinding:
    expected = [
        HANDOFF_RELEASE_BUNDLE_NAME,
        HANDOFF_RELEASE_BUNDLE_JSON,
        HANDOFF_RELEASE_BUNDLE_MD,
    ]
    missing = [name for name in expected if not (release_dir / name).exists()]
    if missing:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="bundle_sidecars",
            title="Bundle Sidecar Artifacts",
            status="failed",
            summary="Bundle artifacts are missing: " + ", ".join(missing),
            required_action="Restore or regenerate the bundle and sidecar manifests.",
            evidence_artifacts=[_release_rel_path(release_id, name) for name in expected],
            metadata={"missing": missing},
        )
    return HandoffReleaseBundleVerificationFinding(
        finding_id="bundle_sidecars",
        title="Bundle Sidecar Artifacts",
        status="passed",
        summary="Bundle ZIP and sidecar manifests exist.",
        evidence_artifacts=[_release_rel_path(release_id, name) for name in expected],
    )


def _archive_hash_finding(
    bundle_path: Path,
    bundle: HandoffReleaseBundleManifest,
    *,
    required: bool,
) -> HandoffReleaseBundleVerificationFinding:
    actual_hash = _file_sha256_or_none(bundle_path)
    metadata = {
        "expected_archive_sha256": bundle.archive_sha256 or None,
        "actual_archive_sha256": actual_hash,
        "expected_archive_size_bytes": bundle.archive_size_bytes,
        "actual_archive_size_bytes": bundle_path.stat().st_size if bundle_path.exists() else None,
        "required": required,
    }
    if not bundle.archive_sha256:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="archive_hash",
            title="Bundle Archive Hash",
            status="failed" if required else "warning",
            summary="Bundle manifest does not record an archive SHA-256.",
            required_action="Regenerate the bundle sidecar manifest from the transfer archive.",
            evidence_artifacts=[bundle.archive_path],
            metadata=metadata,
        )
    if actual_hash != bundle.archive_sha256:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="archive_hash",
            title="Bundle Archive Hash",
            status="failed" if required else "warning",
            summary="Bundle archive SHA-256 does not match the sidecar manifest.",
            required_action="Restore the original ZIP or recreate the bundle from current inputs.",
            evidence_artifacts=[bundle.archive_path],
            metadata=metadata,
        )
    if bundle_path.stat().st_size != bundle.archive_size_bytes:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="archive_hash",
            title="Bundle Archive Hash",
            status="failed" if required else "warning",
            summary="Bundle archive size does not match the sidecar manifest.",
            required_action="Restore the original ZIP or recreate the bundle from current inputs.",
            evidence_artifacts=[bundle.archive_path],
            metadata=metadata,
        )
    return HandoffReleaseBundleVerificationFinding(
        finding_id="archive_hash",
        title="Bundle Archive Hash",
        status="passed",
        summary="Bundle archive hash and size match the sidecar manifest.",
        evidence_artifacts=[bundle.archive_path],
        metadata=metadata,
    )


def _zip_inventory_findings(
    bundle_path: Path,
    bundle: HandoffReleaseBundleManifest,
    *,
    require_manifest_entries: bool,
    require_path_safety: bool,
) -> tuple[
    list[HandoffReleaseBundleVerificationFinding],
    list[HandoffReleaseBundleEntryVerification],
]:
    findings: list[HandoffReleaseBundleVerificationFinding] = []
    entry_verifications: list[HandoffReleaseBundleEntryVerification] = []
    try:
        with zipfile.ZipFile(bundle_path, "r") as archive:
            bad_file = archive.testzip()
            infos = archive.infolist()
            names = [info.filename for info in infos]
            name_counts = {name: names.count(name) for name in set(names)}
            unsafe_names = [name for name in names if not _safe_zip_path(name)]
            duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
            expected_paths = [entry.archive_path for entry in bundle.entries] + [
                HANDOFF_RELEASE_BUNDLE_JSON,
                HANDOFF_RELEASE_BUNDLE_MD,
            ]
            missing_paths = [name for name in expected_paths if name not in name_counts]
            extra_paths = sorted(name for name in name_counts if name not in set(expected_paths))
            embedded_manifest_status = _embedded_manifest_status(archive, bundle)
            for entry in bundle.entries:
                entry_verifications.append(_entry_verification(archive, name_counts, entry))
    except zipfile.BadZipFile as e:
        return [
            HandoffReleaseBundleVerificationFinding(
                finding_id="zip_readable",
                title="ZIP Readability",
                status="failed",
                summary=f"Bundle archive is not a readable ZIP file: {e}",
                required_action="Restore the original ZIP or recreate the bundle.",
                evidence_artifacts=[bundle.archive_path],
            )
        ], []

    if bad_file:
        findings.append(
            HandoffReleaseBundleVerificationFinding(
                finding_id="zip_integrity",
                title="ZIP Internal Integrity",
                status="failed",
                summary=f"ZIP internal CRC check failed for `{bad_file}`.",
                required_action="Restore the original ZIP or recreate the bundle.",
                evidence_artifacts=[bundle.archive_path],
            )
        )
    else:
        findings.append(
            HandoffReleaseBundleVerificationFinding(
                finding_id="zip_integrity",
                title="ZIP Internal Integrity",
                status="passed",
                summary="ZIP internal CRC checks passed.",
                evidence_artifacts=[bundle.archive_path],
            )
        )

    path_status: BundleVerificationStatus = "passed"
    path_summary = "All ZIP entry paths are relative, unique, and traversal-safe."
    path_action = ""
    if unsafe_names or duplicate_names:
        path_status = "failed" if require_path_safety else "warning"
        path_summary = "ZIP contains unsafe or duplicate entry paths."
        path_action = "Recreate the bundle and reject this ZIP for transfer."
    findings.append(
        HandoffReleaseBundleVerificationFinding(
            finding_id="zip_path_safety",
            title="ZIP Path Safety",
            status=path_status,
            summary=path_summary,
            required_action=path_action,
            evidence_artifacts=[bundle.archive_path],
            metadata={
                "unsafe_names": unsafe_names,
                "duplicate_names": duplicate_names,
                "required": require_path_safety,
            },
        )
    )

    failed_entries = [entry for entry in entry_verifications if entry.status == "failed"]
    warning_entries = [entry for entry in entry_verifications if entry.status == "warning"]
    manifest_status: BundleVerificationStatus = "passed"
    manifest_summary = "Every manifest entry is present in the ZIP with matching size and hash."
    manifest_action = ""
    if missing_paths or failed_entries:
        manifest_status = "failed" if require_manifest_entries else "warning"
        manifest_summary = "One or more manifest entries are missing or have changed in the ZIP."
        manifest_action = "Restore the original ZIP or recreate the bundle from current inputs."
    elif warning_entries or extra_paths:
        manifest_status = "warning"
        manifest_summary = "ZIP inventory has extra entries or non-blocking entry warnings."
        manifest_action = "Review unexpected archive entries before transfer."
    findings.append(
        HandoffReleaseBundleVerificationFinding(
            finding_id="manifest_entries",
            title="Manifest Entry Inventory",
            status=manifest_status,
            summary=manifest_summary,
            required_action=manifest_action,
            evidence_artifacts=[bundle.archive_path, HANDOFF_RELEASE_BUNDLE_JSON],
            metadata={
                "expected_entry_count": len(expected_paths),
                "actual_entry_count": len(names),
                "missing_paths": missing_paths,
                "extra_paths": extra_paths,
                "failed_entries": [entry.archive_path for entry in failed_entries],
                "required": require_manifest_entries,
            },
        )
    )
    findings.append(embedded_manifest_status)
    return findings, entry_verifications


def _entry_verification(
    archive: zipfile.ZipFile,
    name_counts: dict[str, int],
    entry: Any,
) -> HandoffReleaseBundleEntryVerification:
    if entry.archive_path not in name_counts:
        return HandoffReleaseBundleEntryVerification(
            archive_path=entry.archive_path,
            entry_type=entry.entry_type,
            expected_sha256=entry.sha256,
            expected_size_bytes=entry.size_bytes,
            status="failed",
            summary="Entry is missing from the ZIP archive.",
        )
    payload = archive.read(entry.archive_path)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    actual_size = len(payload)
    if actual_sha256 != entry.sha256 or actual_size != entry.size_bytes:
        return HandoffReleaseBundleEntryVerification(
            archive_path=entry.archive_path,
            entry_type=entry.entry_type,
            expected_sha256=entry.sha256,
            actual_sha256=actual_sha256,
            expected_size_bytes=entry.size_bytes,
            actual_size_bytes=actual_size,
            status="failed",
            summary="Entry size or SHA-256 does not match the bundle manifest.",
        )
    return HandoffReleaseBundleEntryVerification(
        archive_path=entry.archive_path,
        entry_type=entry.entry_type,
        expected_sha256=entry.sha256,
        actual_sha256=actual_sha256,
        expected_size_bytes=entry.size_bytes,
        actual_size_bytes=actual_size,
        status="passed",
        summary="Entry matches the bundle manifest.",
    )


def _embedded_manifest_status(
    archive: zipfile.ZipFile,
    bundle: HandoffReleaseBundleManifest,
) -> HandoffReleaseBundleVerificationFinding:
    if HANDOFF_RELEASE_BUNDLE_JSON not in archive.namelist():
        return HandoffReleaseBundleVerificationFinding(
            finding_id="embedded_manifest",
            title="Embedded Bundle Manifest",
            status="failed",
            summary="Embedded bundle manifest is missing from the ZIP.",
            required_action="Recreate the bundle.",
            evidence_artifacts=[HANDOFF_RELEASE_BUNDLE_JSON],
        )
    try:
        embedded = json.loads(archive.read(HANDOFF_RELEASE_BUNDLE_JSON).decode("utf-8"))
    except Exception as e:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="embedded_manifest",
            title="Embedded Bundle Manifest",
            status="failed",
            summary=f"Embedded bundle manifest could not be parsed: {type(e).__name__}: {e}",
            required_action="Recreate the bundle.",
            evidence_artifacts=[HANDOFF_RELEASE_BUNDLE_JSON],
        )
    metadata = {
        "embedded_release_id": embedded.get("release_id"),
        "sidecar_release_id": bundle.release_id,
        "embedded_generated_at": embedded.get("generated_at"),
        "sidecar_generated_at": bundle.generated_at,
        "embedded_entry_count": len(embedded.get("entries") or []),
        "sidecar_entry_count": len(bundle.entries),
    }
    if embedded.get("release_id") != bundle.release_id:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="embedded_manifest",
            title="Embedded Bundle Manifest",
            status="failed",
            summary="Embedded bundle manifest release ID does not match the sidecar manifest.",
            required_action="Reject the ZIP or restore the matching sidecar manifest.",
            evidence_artifacts=[HANDOFF_RELEASE_BUNDLE_JSON],
            metadata=metadata,
        )
    if embedded.get("generated_at") != bundle.generated_at:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="embedded_manifest",
            title="Embedded Bundle Manifest",
            status="warning",
            summary="Embedded bundle manifest generation timestamp differs from the sidecar.",
            required_action="Confirm the ZIP and sidecar manifest were produced together.",
            evidence_artifacts=[HANDOFF_RELEASE_BUNDLE_JSON],
            metadata=metadata,
        )
    return HandoffReleaseBundleVerificationFinding(
        finding_id="embedded_manifest",
        title="Embedded Bundle Manifest",
        status="passed",
        summary="Embedded bundle manifest matches the sidecar release identity.",
        evidence_artifacts=[HANDOFF_RELEASE_BUNDLE_JSON],
        metadata=metadata,
    )


def _operator_audit_finding(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleaseBundleVerificationFinding:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleaseBundleVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before release bundle reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleaseBundleVerificationFinding(
        finding_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _safe_zip_path(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute():
        return False
    return ".." not in path.parts


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
