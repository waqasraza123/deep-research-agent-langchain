"""Repository-level custody ledger for handoff release transfer packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release import HandoffReleaseManifest, list_handoff_release_manifests
from .handoff_release_bundle import (
    handoff_release_bundle_path,
    read_handoff_release_bundle_manifest,
)
from .handoff_release_bundle_verification import (
    read_handoff_release_bundle_verification_report,
)
from .handoff_release_receipt import read_handoff_release_receipt
from .handoff_release_verification import read_handoff_release_verification_report
from .operator_audit import verify_operator_audit

ReleaseLedgerReadiness = Literal["complete", "needs_attention", "blocked"]

HANDOFF_LEDGER_DIR = "_handoff"
HANDOFF_RELEASE_LEDGER_JSON = "handoff_release_ledger.json"
HANDOFF_RELEASE_LEDGER_MD = "handoff_release_ledger.md"


class HandoffReleaseLedgerRequest(BaseModel):
    requested_by: str = "operator"
    include_releases_without_receipt: bool = True
    require_release_ready: bool = True
    require_release_verification_valid: bool = True
    require_bundle_ready: bool = True
    require_bundle_verification_valid: bool = True
    require_receipt_recorded: bool = True
    require_recipient_checksum_match: bool = True
    require_global_operator_audit: bool = True
    max_releases: int = Field(default=1000, ge=1, le=10000)
    notes: str = ""


class HandoffReleaseLedgerItem(BaseModel):
    release_id: str
    readiness: ReleaseLedgerReadiness = "needs_attention"
    release_readiness: str = "missing"
    release_generated_at: str = ""
    release_selected_runs: int = 0
    release_verification_readiness: str = "missing"
    release_verification_generated_at: str = ""
    bundle_readiness: str = "missing"
    bundle_generated_at: str = ""
    bundle_archive_sha256: str | None = None
    bundle_archive_valid: bool | None = None
    bundle_verification_readiness: str = "missing"
    bundle_verification_generated_at: str = ""
    receipt_readiness: str = "missing"
    receipt_generated_at: str = ""
    receipt_outcome: str = "missing"
    recipient: str = ""
    transfer_method: str = ""
    transfer_reference: str = ""
    transferred_at: str = ""
    received_at: str = ""
    recipient_bundle_sha256: str | None = None
    recipient_checksum_valid: bool | None = None
    operator_audit_valid: bool = True
    operator_audit_event_count: int = 0
    missing_controls: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class HandoffReleaseLedgerSummary(BaseModel):
    total_releases: int = 0
    indexed_releases: int = 0
    excluded_releases: int = 0
    complete: int = 0
    needs_attention: int = 0
    blocked: int = 0
    missing_receipts: int = 0
    missing_bundles: int = 0
    invalid_bundle_hashes: int = 0
    invalid_recipient_checksums: int = 0
    operator_audit_invalid: int = 0


class HandoffReleaseLedger(BaseModel):
    ledger_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    summary: HandoffReleaseLedgerSummary
    items: list[HandoffReleaseLedgerItem] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    notes: str = ""


def build_handoff_release_ledger(
    *,
    runs_dir: Path,
    request: HandoffReleaseLedgerRequest | None = None,
    releases: list[HandoffReleaseManifest] | None = None,
) -> HandoffReleaseLedger:
    request = request or HandoffReleaseLedgerRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    all_releases = releases if releases is not None else list_handoff_release_manifests(runs_dir)
    limited_releases = all_releases[: request.max_releases]
    warnings: list[str] = []
    if len(all_releases) > request.max_releases:
        warnings.append(
            "Ledger indexed "
            f"{request.max_releases} of {len(all_releases)} releases because max_releases "
            "was reached."
        )

    audit_valid, audit_event_count, audit_warning = _global_audit_state(runs_dir)
    if audit_warning:
        warnings.append(audit_warning)

    items: list[HandoffReleaseLedgerItem] = []
    excluded = len(all_releases) - len(limited_releases)
    for release in limited_releases:
        item = _ledger_item(
            runs_dir,
            release,
            request=request,
            operator_audit_valid=audit_valid,
            operator_audit_event_count=audit_event_count,
        )
        if item.receipt_readiness == "missing" and not request.include_releases_without_receipt:
            excluded += 1
            continue
        items.append(item)

    summary = HandoffReleaseLedgerSummary(
        total_releases=len(all_releases),
        indexed_releases=len(items),
        excluded_releases=excluded,
        complete=sum(1 for item in items if item.readiness == "complete"),
        needs_attention=sum(1 for item in items if item.readiness == "needs_attention"),
        blocked=sum(1 for item in items if item.readiness == "blocked"),
        missing_receipts=sum(1 for item in items if item.receipt_readiness == "missing"),
        missing_bundles=sum(1 for item in items if item.bundle_readiness == "missing"),
        invalid_bundle_hashes=sum(1 for item in items if item.bundle_archive_valid is False),
        invalid_recipient_checksums=sum(
            1 for item in items if item.recipient_checksum_valid is False
        ),
        operator_audit_invalid=sum(1 for item in items if not item.operator_audit_valid),
    )
    ledger = HandoffReleaseLedger(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        summary=summary,
        items=items,
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_MD}",
        ],
        warnings=warnings,
        required_controls={
            "release_ready": request.require_release_ready,
            "release_verification_valid": request.require_release_verification_valid,
            "bundle_ready": request.require_bundle_ready,
            "bundle_verification_valid": request.require_bundle_verification_valid,
            "receipt_recorded": request.require_receipt_recorded,
            "recipient_checksum_match": request.require_recipient_checksum_match,
            "global_operator_audit": request.require_global_operator_audit,
        },
        notes=request.notes,
    )
    write_handoff_release_ledger(runs_dir, ledger)
    return ledger


def read_handoff_release_ledger(runs_dir: Path) -> HandoffReleaseLedger:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_LEDGER_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseLedger, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseLedger.parse_obj(data)


def write_handoff_release_ledger(
    runs_dir: Path,
    ledger: HandoffReleaseLedger,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_LEDGER_JSON).write_text(
        json.dumps(_model_to_plain(ledger), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_LEDGER_MD).write_text(
        render_handoff_release_ledger_markdown(ledger),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_MD}",
    ]


def render_handoff_release_ledger_markdown(ledger: HandoffReleaseLedger) -> str:
    lines = [
        "# Handoff Release Ledger",
        "",
        f"- Generated at: `{ledger.generated_at}`",
        f"- Requested by: `{ledger.requested_by}`",
        f"- Total releases: {ledger.summary.total_releases}",
        f"- Indexed releases: {ledger.summary.indexed_releases}",
        f"- Complete: {ledger.summary.complete}",
        f"- Needs attention: {ledger.summary.needs_attention}",
        f"- Blocked: {ledger.summary.blocked}",
        f"- Missing receipts: {ledger.summary.missing_receipts}",
        f"- Missing bundles: {ledger.summary.missing_bundles}",
        f"- Invalid bundle hashes: {ledger.summary.invalid_bundle_hashes}",
        f"- Invalid recipient checksums: {ledger.summary.invalid_recipient_checksums}",
        f"- Invalid operator audit: {ledger.summary.operator_audit_invalid}",
        "",
        "## Releases",
        "",
    ]
    if not ledger.items:
        lines.extend(["No releases were indexed.", ""])
    for item in ledger.items:
        lines.extend(
            [
                f"### {item.release_id}",
                "",
                f"- Readiness: `{item.readiness}`",
                f"- Release readiness: `{item.release_readiness}`",
                f"- Release verification: `{item.release_verification_readiness}`",
                f"- Bundle readiness: `{item.bundle_readiness}`",
                f"- Bundle verification: `{item.bundle_verification_readiness}`",
                f"- Bundle hash valid: `{item.bundle_archive_valid}`",
                f"- Receipt readiness: `{item.receipt_readiness}`",
                f"- Receipt outcome: `{item.receipt_outcome}`",
                f"- Recipient: {item.recipient or 'unspecified'}",
                f"- Transfer method: {item.transfer_method or 'unspecified'}",
                f"- Transfer reference: {item.transfer_reference or 'unspecified'}",
                f"- Recipient checksum valid: `{item.recipient_checksum_valid}`",
                f"- Operator audit valid: `{item.operator_audit_valid}`",
                "- Missing controls: "
                + (", ".join(f"`{control}`" for control in item.missing_controls) or "none"),
                "- Blockers: " + (", ".join(item.blockers) or "none"),
                "- Warnings: " + (", ".join(item.warnings) or "none"),
                "",
            ]
        )
    if ledger.warnings:
        lines.extend(["## Ledger Warnings", ""])
        lines.extend(f"- {warning}" for warning in ledger.warnings)
        lines.append("")
    if ledger.notes:
        lines.extend(["## Notes", "", ledger.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _ledger_item(
    runs_dir: Path,
    release: HandoffReleaseManifest,
    *,
    request: HandoffReleaseLedgerRequest,
    operator_audit_valid: bool,
    operator_audit_event_count: int,
) -> HandoffReleaseLedgerItem:
    release_id = release.release_id
    missing_controls: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    artifacts = [
        f"_handoff/releases/{release_id}/handoff_release.json",
        f"_handoff/releases/{release_id}/handoff_release.md",
    ]

    if request.require_release_ready and release.readiness != "ready_for_release":
        blockers.append(f"Release readiness is `{release.readiness}`.")
    elif release.readiness != "ready_for_release":
        warnings.append(f"Release readiness is `{release.readiness}`.")

    release_verification_readiness = "missing"
    release_verification_generated_at = ""
    try:
        release_verification = read_handoff_release_verification_report(runs_dir, release_id)
        release_verification_readiness = release_verification.readiness
        release_verification_generated_at = release_verification.generated_at
        artifacts.extend(release_verification.artifacts)
        if (
            request.require_release_verification_valid
            and release_verification.readiness != "valid"
        ):
            blockers.append(
                f"Release verification readiness is `{release_verification.readiness}`."
            )
        elif release_verification.readiness != "valid":
            warnings.append(
                f"Release verification readiness is `{release_verification.readiness}`."
            )
    except FileNotFoundError:
        missing_controls.append("handoff_release_verification")
        message = "Release verification report is missing."
        if request.require_release_verification_valid:
            blockers.append(message)
        else:
            warnings.append(message)

    bundle_readiness = "missing"
    bundle_generated_at = ""
    bundle_archive_sha256: str | None = None
    bundle_archive_valid: bool | None = None
    try:
        bundle = read_handoff_release_bundle_manifest(runs_dir, release_id)
        bundle_readiness = bundle.readiness
        bundle_generated_at = bundle.generated_at
        artifacts.extend(
            [
                f"_handoff/releases/{release_id}/handoff_release_bundle.zip",
                f"_handoff/releases/{release_id}/handoff_release_bundle_manifest.json",
                f"_handoff/releases/{release_id}/handoff_release_bundle_manifest.md",
            ]
        )
        bundle_archive_sha256, bundle_archive_valid = _bundle_hash_state(runs_dir, bundle)
        if request.require_bundle_ready and bundle.readiness != "ready":
            blockers.append(f"Bundle readiness is `{bundle.readiness}`.")
        elif bundle.readiness != "ready":
            warnings.append(f"Bundle readiness is `{bundle.readiness}`.")
        if bundle_archive_valid is False:
            blockers.append("Bundle archive hash does not match its sidecar manifest.")
        elif bundle_archive_valid is None:
            warnings.append("Bundle sidecar manifest does not record an archive SHA-256.")
    except FileNotFoundError:
        missing_controls.append("handoff_release_bundle")
        message = "Release bundle is missing."
        if request.require_bundle_ready:
            blockers.append(message)
        else:
            warnings.append(message)

    bundle_verification_readiness = "missing"
    bundle_verification_generated_at = ""
    try:
        bundle_verification = read_handoff_release_bundle_verification_report(
            runs_dir,
            release_id,
        )
        bundle_verification_readiness = bundle_verification.readiness
        bundle_verification_generated_at = bundle_verification.generated_at
        artifacts.extend(bundle_verification.artifacts)
        if (
            request.require_bundle_verification_valid
            and bundle_verification.readiness != "valid"
        ):
            blockers.append(
                f"Bundle verification readiness is `{bundle_verification.readiness}`."
            )
        elif bundle_verification.readiness != "valid":
            warnings.append(
                f"Bundle verification readiness is `{bundle_verification.readiness}`."
            )
    except FileNotFoundError:
        missing_controls.append("handoff_release_bundle_verification")
        message = "Bundle verification report is missing."
        if request.require_bundle_verification_valid:
            blockers.append(message)
        else:
            warnings.append(message)

    receipt_readiness = "missing"
    receipt_generated_at = ""
    receipt_outcome = "missing"
    recipient = ""
    transfer_method = ""
    transfer_reference = ""
    transferred_at = ""
    received_at = ""
    recipient_bundle_sha256: str | None = None
    recipient_checksum_valid: bool | None = None
    try:
        receipt = read_handoff_release_receipt(runs_dir, release_id)
        receipt_readiness = receipt.readiness
        receipt_generated_at = receipt.generated_at
        receipt_outcome = receipt.outcome
        recipient = receipt.recipient
        transfer_method = receipt.transfer_method
        transfer_reference = receipt.transfer_reference
        transferred_at = receipt.transferred_at
        received_at = receipt.received_at
        recipient_bundle_sha256 = receipt.recipient_bundle_sha256
        recipient_checksum_valid = _recipient_checksum_valid(receipt)
        artifacts.extend(receipt.artifacts)
        if request.require_receipt_recorded and receipt.readiness != "recorded":
            blockers.append(f"Receipt readiness is `{receipt.readiness}`.")
        elif receipt.readiness != "recorded":
            warnings.append(f"Receipt readiness is `{receipt.readiness}`.")
        if request.require_recipient_checksum_match and recipient_checksum_valid is not True:
            blockers.append("Recipient checksum is missing or does not match the bundle manifest.")
        elif recipient_checksum_valid is not True:
            warnings.append("Recipient checksum is missing or does not match the bundle manifest.")
    except FileNotFoundError:
        missing_controls.append("handoff_release_receipt")
        message = "Release transfer receipt is missing."
        if request.require_receipt_recorded:
            blockers.append(message)
        else:
            warnings.append(message)

    if request.require_global_operator_audit and not operator_audit_valid:
        blockers.append("Global operator audit hash-chain verification failed.")
    elif not operator_audit_valid:
        warnings.append("Global operator audit hash-chain verification failed.")

    readiness: ReleaseLedgerReadiness = "complete"
    blockers = _dedupe(blockers)
    warnings = _dedupe(warnings)
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "needs_attention"

    return HandoffReleaseLedgerItem(
        release_id=release_id,
        readiness=readiness,
        release_readiness=release.readiness,
        release_generated_at=release.generated_at,
        release_selected_runs=len(release.runs),
        release_verification_readiness=release_verification_readiness,
        release_verification_generated_at=release_verification_generated_at,
        bundle_readiness=bundle_readiness,
        bundle_generated_at=bundle_generated_at,
        bundle_archive_sha256=bundle_archive_sha256,
        bundle_archive_valid=bundle_archive_valid,
        bundle_verification_readiness=bundle_verification_readiness,
        bundle_verification_generated_at=bundle_verification_generated_at,
        receipt_readiness=receipt_readiness,
        receipt_generated_at=receipt_generated_at,
        receipt_outcome=receipt_outcome,
        recipient=recipient,
        transfer_method=transfer_method,
        transfer_reference=transfer_reference,
        transferred_at=transferred_at,
        received_at=received_at,
        recipient_bundle_sha256=recipient_bundle_sha256,
        recipient_checksum_valid=recipient_checksum_valid,
        operator_audit_valid=operator_audit_valid,
        operator_audit_event_count=operator_audit_event_count,
        missing_controls=_dedupe(missing_controls),
        blockers=blockers,
        warnings=warnings,
        artifacts=_dedupe(artifacts),
    )


def _bundle_hash_state(runs_dir: Path, bundle: Any) -> tuple[str | None, bool | None]:
    try:
        path = handoff_release_bundle_path(runs_dir, bundle.release_id)
    except FileNotFoundError:
        return None, False
    actual_hash = file_sha256(path)
    if not bundle.archive_sha256:
        return actual_hash, None
    return actual_hash, actual_hash == bundle.archive_sha256


def _recipient_checksum_valid(receipt: Any) -> bool | None:
    if not receipt.recipient_bundle_sha256:
        return False
    if not receipt.expected_bundle_archive_sha256:
        return None
    return receipt.recipient_bundle_sha256 == receipt.expected_bundle_archive_sha256.lower()


def _global_audit_state(runs_dir: Path) -> tuple[bool, int, str | None]:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return False, 0, f"Global operator audit could not be verified: {e}"
    return verification.valid, verification.event_count, None


def _ledger_dir(runs_dir: Path) -> Path:
    root = runs_dir.resolve()
    ledger_dir = (root / HANDOFF_LEDGER_DIR).resolve()
    if root != ledger_dir and root not in ledger_dir.parents:
        raise ValueError("Invalid handoff release ledger directory")
    return ledger_dir


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
