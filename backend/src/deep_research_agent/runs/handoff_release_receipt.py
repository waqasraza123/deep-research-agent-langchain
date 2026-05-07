"""Transfer receipt records for portable handoff release bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release import HANDOFF_RELEASES_DIR, read_handoff_release_manifest
from .handoff_release_bundle import (
    HANDOFF_RELEASE_BUNDLE_NAME,
    HandoffReleaseBundleManifest,
    handoff_release_bundle_path,
    read_handoff_release_bundle_manifest,
)
from .handoff_release_bundle_verification import (
    HandoffReleaseBundleVerificationReport,
    read_handoff_release_bundle_verification_report,
)
from .operator_audit import verify_operator_audit

ReleaseReceiptReadiness = Literal["recorded", "warnings", "blocked"]
ReleaseReceiptOutcome = Literal["accepted", "accepted_with_exceptions", "rejected", "pending"]

HANDOFF_RELEASE_RECEIPT_JSON = "handoff_release_receipt.json"
HANDOFF_RELEASE_RECEIPT_MD = "handoff_release_receipt.md"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class HandoffReleaseReceiptRequest(BaseModel):
    requested_by: str = "operator"
    recipient: str = ""
    recipient_contact: str = ""
    transfer_method: str = ""
    transfer_reference: str = ""
    transferred_at: str | None = None
    received_by: str = ""
    received_at: str | None = None
    recipient_bundle_sha256: str | None = None
    outcome: ReleaseReceiptOutcome = "accepted"
    require_bundle_ready: bool = True
    require_bundle_verification_valid: bool = True
    require_bundle_hash_match: bool = True
    require_recipient_checksum_match: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleaseReceiptCheck(BaseModel):
    check_id: str
    title: str
    status: Literal["passed", "warning", "failed"]
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleaseReceiptSummary(BaseModel):
    selected_runs: int = 0
    included_run_exports: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    checks_passed: int = 0
    checks_warning: int = 0
    checks_failed: int = 0


class HandoffReleaseReceipt(BaseModel):
    receipt_version: str = "1.0"
    release_id: str
    generated_at: str
    requested_by: str = "operator"
    recipient: str = ""
    recipient_contact: str = ""
    transfer_method: str = ""
    transfer_reference: str = ""
    transferred_at: str
    received_by: str = ""
    received_at: str = ""
    recipient_bundle_sha256: str | None = None
    outcome: ReleaseReceiptOutcome = "accepted"
    readiness: ReleaseReceiptReadiness = "warnings"
    release_readiness: str = "missing"
    bundle_readiness: str = "missing"
    bundle_verification_readiness: str = "missing"
    bundle_archive_path: str = ""
    bundle_archive_sha256: str | None = None
    expected_bundle_archive_sha256: str | None = None
    release_generated_at: str = ""
    bundle_generated_at: str = ""
    bundle_verification_generated_at: str = ""
    checks: list[HandoffReleaseReceiptCheck] = Field(default_factory=list)
    summary: HandoffReleaseReceiptSummary = Field(default_factory=HandoffReleaseReceiptSummary)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_receipt(
    *,
    runs_dir: Path,
    release_id: str,
    request: HandoffReleaseReceiptRequest | None = None,
) -> HandoffReleaseReceipt:
    request = request or HandoffReleaseReceiptRequest()
    release = read_handoff_release_manifest(runs_dir, release_id)
    bundle = read_handoff_release_bundle_manifest(runs_dir, release.release_id)
    bundle_path = _release_dir(runs_dir, release.release_id) / HANDOFF_RELEASE_BUNDLE_NAME
    try:
        verification = read_handoff_release_bundle_verification_report(
            runs_dir,
            release.release_id,
        )
    except FileNotFoundError:
        verification = None

    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    transferred_at = request.transferred_at or now_iso_utc()
    received_at = request.received_at or ""
    checks = _receipt_checks(
        runs_dir,
        request=request,
        release_readiness=release.readiness,
        bundle=bundle,
        verification=verification,
    )
    blockers = [check.summary for check in checks if check.status == "failed"]
    warnings = [check.summary for check in checks if check.status == "warning"]
    readiness: ReleaseReceiptReadiness = "recorded"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "warnings"

    receipt = HandoffReleaseReceipt(
        release_id=release.release_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        recipient=request.recipient,
        recipient_contact=request.recipient_contact,
        transfer_method=request.transfer_method,
        transfer_reference=request.transfer_reference,
        transferred_at=transferred_at,
        received_by=request.received_by,
        received_at=received_at,
        recipient_bundle_sha256=_normalize_optional_hash(request.recipient_bundle_sha256),
        outcome=request.outcome,
        readiness=readiness,
        release_readiness=release.readiness,
        bundle_readiness=bundle.readiness,
        bundle_verification_readiness=verification.readiness if verification else "missing",
        bundle_archive_path=bundle.archive_path,
        bundle_archive_sha256=_file_sha256_or_none(bundle_path),
        expected_bundle_archive_sha256=bundle.archive_sha256 or None,
        release_generated_at=release.generated_at,
        bundle_generated_at=bundle.generated_at,
        bundle_verification_generated_at=verification.generated_at if verification else "",
        checks=checks,
        summary=HandoffReleaseReceiptSummary(
            selected_runs=len(release.runs),
            included_run_exports=bundle.summary.included_run_exports,
            blocker_count=len(blockers),
            warning_count=len(warnings),
            checks_passed=sum(1 for check in checks if check.status == "passed"),
            checks_warning=sum(1 for check in checks if check.status == "warning"),
            checks_failed=sum(1 for check in checks if check.status == "failed"),
        ),
        blockers=_dedupe(blockers),
        warnings=_dedupe(warnings),
        required_controls={
            "bundle_ready": request.require_bundle_ready,
            "bundle_verification_valid": request.require_bundle_verification_valid,
            "bundle_hash_match": request.require_bundle_hash_match,
            "recipient_checksum_match": request.require_recipient_checksum_match,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            _release_rel_path(release.release_id, HANDOFF_RELEASE_RECEIPT_JSON),
            _release_rel_path(release.release_id, HANDOFF_RELEASE_RECEIPT_MD),
        ],
        notes=request.notes,
    )
    write_handoff_release_receipt(runs_dir, receipt)
    return receipt


def read_handoff_release_receipt(runs_dir: Path, release_id: str) -> HandoffReleaseReceipt:
    path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_RECEIPT_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseReceipt, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseReceipt.parse_obj(data)


def write_handoff_release_receipt(
    runs_dir: Path,
    receipt: HandoffReleaseReceipt,
) -> list[str]:
    release_dir = _release_dir(runs_dir, receipt.release_id)
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / HANDOFF_RELEASE_RECEIPT_JSON).write_text(
        json.dumps(_model_to_plain(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / HANDOFF_RELEASE_RECEIPT_MD).write_text(
        render_handoff_release_receipt_markdown(receipt),
        encoding="utf-8",
    )
    return [
        _release_rel_path(receipt.release_id, HANDOFF_RELEASE_RECEIPT_JSON),
        _release_rel_path(receipt.release_id, HANDOFF_RELEASE_RECEIPT_MD),
    ]


def render_handoff_release_receipt_markdown(receipt: HandoffReleaseReceipt) -> str:
    lines = [
        "# Handoff Release Receipt",
        "",
        f"- Release ID: `{receipt.release_id}`",
        f"- Generated at: `{receipt.generated_at}`",
        f"- Requested by: `{receipt.requested_by}`",
        f"- Recipient: {receipt.recipient or 'unspecified'}",
        f"- Recipient contact: {receipt.recipient_contact or 'unspecified'}",
        f"- Transfer method: {receipt.transfer_method or 'unspecified'}",
        f"- Transfer reference: {receipt.transfer_reference or 'unspecified'}",
        f"- Transferred at: `{receipt.transferred_at}`",
        f"- Received by: {receipt.received_by or 'unspecified'}",
        f"- Received at: `{receipt.received_at or 'unconfirmed'}`",
        f"- Outcome: `{receipt.outcome}`",
        f"- Readiness: `{receipt.readiness}`",
        f"- Release readiness: `{receipt.release_readiness}`",
        f"- Bundle readiness: `{receipt.bundle_readiness}`",
        f"- Bundle verification readiness: `{receipt.bundle_verification_readiness}`",
        f"- Bundle archive SHA-256: `{receipt.bundle_archive_sha256 or 'unavailable'}`",
        f"- Recipient bundle SHA-256: `{receipt.recipient_bundle_sha256 or 'unavailable'}`",
        "",
        "## Checks",
        "",
    ]
    for check in receipt.checks:
        lines.extend(
            [
                f"### {check.title}",
                "",
                f"- Status: `{check.status}`",
                f"- Summary: {check.summary or 'None'}",
                f"- Required action: {check.required_action or 'None'}",
                "- Evidence: "
                + (", ".join(f"`{item}`" for item in check.evidence_artifacts) or "none"),
                "",
            ]
        )
    if receipt.blockers:
        lines.extend(["## Receipt Blockers", ""])
        lines.extend(f"- {item}" for item in receipt.blockers)
        lines.append("")
    if receipt.warnings:
        lines.extend(["## Receipt Warnings", ""])
        lines.extend(f"- {item}" for item in receipt.warnings)
        lines.append("")
    if receipt.notes:
        lines.extend(["## Notes", "", receipt.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _receipt_checks(
    runs_dir: Path,
    *,
    request: HandoffReleaseReceiptRequest,
    release_readiness: str,
    bundle: HandoffReleaseBundleManifest,
    verification: HandoffReleaseBundleVerificationReport | None,
) -> list[HandoffReleaseReceiptCheck]:
    return [
        _release_ready_check(release_readiness),
        _bundle_ready_check(bundle, required=request.require_bundle_ready),
        _verification_ready_check(
            verification,
            required=request.require_bundle_verification_valid,
        ),
        _bundle_hash_check(
            runs_dir,
            bundle,
            required=request.require_bundle_hash_match,
        ),
        _recipient_checksum_check(
            bundle,
            request.recipient_bundle_sha256,
            required=request.require_recipient_checksum_match,
        ),
        _transfer_metadata_check(request),
        _recipient_outcome_check(request),
        _operator_audit_check(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]


def _release_ready_check(release_readiness: str) -> HandoffReleaseReceiptCheck:
    if release_readiness != "ready_for_release":
        return HandoffReleaseReceiptCheck(
            check_id="release_ready",
            title="Release Readiness",
            status="warning",
            summary=f"Release readiness is `{release_readiness}`.",
            required_action="Confirm exception approval before relying on this transfer receipt.",
            evidence_artifacts=["handoff_release.json"],
            metadata={"release_readiness": release_readiness},
        )
    return HandoffReleaseReceiptCheck(
        check_id="release_ready",
        title="Release Readiness",
        status="passed",
        summary="Release manifest is ready for release.",
        evidence_artifacts=["handoff_release.json"],
        metadata={"release_readiness": release_readiness},
    )


def _bundle_ready_check(
    bundle: HandoffReleaseBundleManifest,
    *,
    required: bool,
) -> HandoffReleaseReceiptCheck:
    if bundle.readiness != "ready":
        return HandoffReleaseReceiptCheck(
            check_id="bundle_ready",
            title="Bundle Readiness",
            status="failed" if required else "warning",
            summary=f"Bundle readiness is `{bundle.readiness}`.",
            required_action="Regenerate or resolve the release bundle before transfer reliance.",
            evidence_artifacts=[bundle.archive_path, "handoff_release_bundle_manifest.json"],
            metadata={"bundle_readiness": bundle.readiness, "required": required},
        )
    return HandoffReleaseReceiptCheck(
        check_id="bundle_ready",
        title="Bundle Readiness",
        status="passed",
        summary="Bundle manifest is ready.",
        evidence_artifacts=[bundle.archive_path, "handoff_release_bundle_manifest.json"],
        metadata={"bundle_readiness": bundle.readiness, "required": required},
    )


def _verification_ready_check(
    verification: HandoffReleaseBundleVerificationReport | None,
    *,
    required: bool,
) -> HandoffReleaseReceiptCheck:
    if verification is None:
        return HandoffReleaseReceiptCheck(
            check_id="bundle_verification",
            title="Bundle Verification",
            status="failed" if required else "warning",
            summary="Bundle verification report is missing.",
            required_action="Generate bundle verification before recording final receipt.",
            evidence_artifacts=["handoff_release_bundle_verification.json"],
            metadata={"required": required},
        )
    if verification.readiness != "valid":
        return HandoffReleaseReceiptCheck(
            check_id="bundle_verification",
            title="Bundle Verification",
            status="failed" if required else "warning",
            summary=f"Bundle verification readiness is `{verification.readiness}`.",
            required_action="Resolve bundle verification failures or document an exception.",
            evidence_artifacts=["handoff_release_bundle_verification.json"],
            metadata={"readiness": verification.readiness, "required": required},
        )
    return HandoffReleaseReceiptCheck(
        check_id="bundle_verification",
        title="Bundle Verification",
        status="passed",
        summary="Bundle verification report is valid.",
        evidence_artifacts=["handoff_release_bundle_verification.json"],
        metadata={"readiness": verification.readiness, "required": required},
    )


def _bundle_hash_check(
    runs_dir: Path,
    bundle: HandoffReleaseBundleManifest,
    *,
    required: bool,
) -> HandoffReleaseReceiptCheck:
    try:
        bundle_path = handoff_release_bundle_path(runs_dir, bundle.release_id)
    except FileNotFoundError:
        return HandoffReleaseReceiptCheck(
            check_id="bundle_hash",
            title="Bundle Archive Hash",
            status="failed" if required else "warning",
            summary="Bundle archive is missing.",
            required_action="Restore or recreate the bundle before transfer reliance.",
            evidence_artifacts=[bundle.archive_path, "handoff_release_bundle_manifest.json"],
            metadata={
                "expected_bundle_sha256": bundle.archive_sha256 or None,
                "actual_bundle_sha256": None,
                "required": required,
            },
        )
    actual_hash = file_sha256(bundle_path)
    metadata = {
        "expected_bundle_sha256": bundle.archive_sha256 or None,
        "actual_bundle_sha256": actual_hash,
        "required": required,
    }
    if not bundle.archive_sha256 or bundle.archive_sha256 != actual_hash:
        return HandoffReleaseReceiptCheck(
            check_id="bundle_hash",
            title="Bundle Archive Hash",
            status="failed" if required else "warning",
            summary="Current bundle archive hash does not match the sidecar manifest.",
            required_action="Restore or recreate the bundle before transfer reliance.",
            evidence_artifacts=[bundle.archive_path, "handoff_release_bundle_manifest.json"],
            metadata=metadata,
        )
    return HandoffReleaseReceiptCheck(
        check_id="bundle_hash",
        title="Bundle Archive Hash",
        status="passed",
        summary="Current bundle archive hash matches the sidecar manifest.",
        evidence_artifacts=[bundle.archive_path, "handoff_release_bundle_manifest.json"],
        metadata=metadata,
    )


def _recipient_checksum_check(
    bundle: HandoffReleaseBundleManifest,
    recipient_hash: str | None,
    *,
    required: bool,
) -> HandoffReleaseReceiptCheck:
    normalized = _normalize_optional_hash(recipient_hash)
    metadata = {
        "expected_bundle_sha256": bundle.archive_sha256 or None,
        "recipient_bundle_sha256": normalized,
        "required": required,
    }
    if not normalized:
        return HandoffReleaseReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient bundle SHA-256 was not provided.",
            required_action="Record the checksum observed by the recipient.",
            evidence_artifacts=[bundle.archive_path],
            metadata=metadata,
        )
    if not SHA256_PATTERN.match(normalized):
        return HandoffReleaseReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient bundle SHA-256 is not a valid 64-character hex digest.",
            required_action="Record the exact SHA-256 digest observed by the recipient.",
            evidence_artifacts=[bundle.archive_path],
            metadata=metadata,
        )
    if bundle.archive_sha256 and normalized != bundle.archive_sha256.lower():
        return HandoffReleaseReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient bundle SHA-256 does not match the bundle manifest.",
            required_action="Do not rely on this transfer until the recipient confirms the hash.",
            evidence_artifacts=[bundle.archive_path],
            metadata=metadata,
        )
    return HandoffReleaseReceiptCheck(
        check_id="recipient_checksum",
        title="Recipient Checksum",
        status="passed",
        summary="Recipient bundle SHA-256 matches the bundle manifest.",
        evidence_artifacts=[bundle.archive_path],
        metadata=metadata,
    )


def _transfer_metadata_check(request: HandoffReleaseReceiptRequest) -> HandoffReleaseReceiptCheck:
    missing = []
    if not request.recipient.strip():
        missing.append("recipient")
    if not request.transfer_method.strip():
        missing.append("transfer_method")
    if not request.transfer_reference.strip():
        missing.append("transfer_reference")
    if missing:
        return HandoffReleaseReceiptCheck(
            check_id="transfer_metadata",
            title="Transfer Metadata",
            status="warning",
            summary="Transfer receipt metadata is incomplete: " + ", ".join(missing),
            required_action="Record recipient, transfer method, and transfer reference.",
            metadata={"missing": missing},
        )
    return HandoffReleaseReceiptCheck(
        check_id="transfer_metadata",
        title="Transfer Metadata",
        status="passed",
        summary="Transfer receipt metadata is complete.",
        metadata={
            "recipient": request.recipient,
            "transfer_method": request.transfer_method,
            "transfer_reference": request.transfer_reference,
        },
    )


def _recipient_outcome_check(request: HandoffReleaseReceiptRequest) -> HandoffReleaseReceiptCheck:
    if request.outcome == "rejected":
        return HandoffReleaseReceiptCheck(
            check_id="recipient_outcome",
            title="Recipient Outcome",
            status="failed",
            summary="Recipient outcome is rejected.",
            required_action=(
                "Do not treat this release as transferred until rejection is resolved."
            ),
            metadata={"outcome": request.outcome},
        )
    if request.outcome in {"accepted_with_exceptions", "pending"}:
        return HandoffReleaseReceiptCheck(
            check_id="recipient_outcome",
            title="Recipient Outcome",
            status="warning",
            summary=f"Recipient outcome is `{request.outcome}`.",
            required_action="Track recipient exceptions or pending confirmation to closure.",
            metadata={"outcome": request.outcome},
        )
    return HandoffReleaseReceiptCheck(
        check_id="recipient_outcome",
        title="Recipient Outcome",
        status="passed",
        summary="Recipient outcome is accepted.",
        metadata={"outcome": request.outcome},
    )


def _operator_audit_check(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleaseReceiptCheck:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleaseReceiptCheck(
            check_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleaseReceiptCheck(
            check_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before receipt reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleaseReceiptCheck(
        check_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _normalize_optional_hash(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return cleaned


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


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = item.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
