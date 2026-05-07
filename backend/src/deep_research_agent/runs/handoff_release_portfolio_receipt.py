"""Portfolio-level receipt records for final handoff release attestations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release_attestation import (
    HANDOFF_RELEASE_ATTESTATION_JSON,
    HandoffReleaseAttestation,
    read_handoff_release_attestation,
)
from .handoff_release_attestation_verification import (
    HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON,
    HandoffReleaseAttestationVerificationReport,
    read_handoff_release_attestation_verification_report,
)
from .handoff_release_ledger import HANDOFF_LEDGER_DIR
from .operator_audit import verify_operator_audit

PortfolioReceiptReadiness = Literal["recorded", "warnings", "blocked"]
PortfolioReceiptOutcome = Literal["accepted", "accepted_with_exceptions", "rejected", "pending"]

HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON = "handoff_release_portfolio_receipt.json"
HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD = "handoff_release_portfolio_receipt.md"
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class HandoffReleasePortfolioReceiptRequest(BaseModel):
    requested_by: str = "operator"
    recipient: str = ""
    recipient_contact: str = ""
    transfer_method: str = ""
    transfer_reference: str = ""
    transferred_at: str | None = None
    received_by: str = ""
    received_at: str | None = None
    recipient_attestation_sha256: str | None = None
    outcome: PortfolioReceiptOutcome = "accepted"
    require_attestation_ready: bool = True
    require_attestation_verification_valid: bool = True
    require_attestation_hash_match: bool = True
    require_recipient_checksum_match: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleasePortfolioReceiptCheck(BaseModel):
    check_id: str
    title: str
    status: Literal["passed", "warning", "failed"]
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleasePortfolioReceiptSummary(BaseModel):
    release_count: int = 0
    attested_artifacts: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    checks_passed: int = 0
    checks_warning: int = 0
    checks_failed: int = 0


class HandoffReleasePortfolioReceipt(BaseModel):
    receipt_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    recipient: str = ""
    recipient_contact: str = ""
    transfer_method: str = ""
    transfer_reference: str = ""
    transferred_at: str
    received_by: str = ""
    received_at: str = ""
    recipient_attestation_sha256: str | None = None
    outcome: PortfolioReceiptOutcome = "accepted"
    readiness: PortfolioReceiptReadiness = "warnings"
    attestation_readiness: str = "missing"
    attestation_verification_readiness: str = "missing"
    attestation_sha256: str | None = None
    expected_attestation_sha256: str | None = None
    attestation_generated_at: str = ""
    attestation_verification_generated_at: str = ""
    checks: list[HandoffReleasePortfolioReceiptCheck] = Field(default_factory=list)
    summary: HandoffReleasePortfolioReceiptSummary = Field(
        default_factory=HandoffReleasePortfolioReceiptSummary
    )
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_portfolio_receipt(
    *,
    runs_dir: Path,
    request: HandoffReleasePortfolioReceiptRequest | None = None,
) -> HandoffReleasePortfolioReceipt:
    request = request or HandoffReleasePortfolioReceiptRequest()
    attestation = read_handoff_release_attestation(runs_dir)
    verification = read_handoff_release_attestation_verification_report(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    transferred_at = request.transferred_at or now_iso_utc()
    received_at = request.received_at or ""
    attestation_path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_ATTESTATION_JSON
    current_attestation_sha = _file_sha256_or_none(attestation_path)

    checks = _receipt_checks(
        runs_dir,
        request=request,
        attestation=attestation,
        verification=verification,
        current_attestation_sha=current_attestation_sha,
    )
    blockers = _dedupe(check.summary for check in checks if check.status == "failed")
    warnings = _dedupe(check.summary for check in checks if check.status == "warning")
    readiness: PortfolioReceiptReadiness = "recorded"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "warnings"

    receipt = HandoffReleasePortfolioReceipt(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        recipient=request.recipient,
        recipient_contact=request.recipient_contact,
        transfer_method=request.transfer_method,
        transfer_reference=request.transfer_reference,
        transferred_at=transferred_at,
        received_by=request.received_by,
        received_at=received_at,
        recipient_attestation_sha256=_normalize_optional_hash(
            request.recipient_attestation_sha256
        ),
        outcome=request.outcome,
        readiness=readiness,
        attestation_readiness=attestation.readiness,
        attestation_verification_readiness=verification.readiness,
        attestation_sha256=current_attestation_sha,
        expected_attestation_sha256=verification.attestation_sha256,
        attestation_generated_at=attestation.generated_at,
        attestation_verification_generated_at=verification.generated_at,
        checks=checks,
        summary=HandoffReleasePortfolioReceiptSummary(
            release_count=attestation.summary.release_count,
            attested_artifacts=attestation.summary.artifact_count,
            blocker_count=len(blockers),
            warning_count=len(warnings),
            checks_passed=sum(1 for check in checks if check.status == "passed"),
            checks_warning=sum(1 for check in checks if check.status == "warning"),
            checks_failed=sum(1 for check in checks if check.status == "failed"),
        ),
        blockers=blockers,
        warnings=warnings,
        required_controls={
            "attestation_ready": request.require_attestation_ready,
            "attestation_verification_valid": (
                request.require_attestation_verification_valid
            ),
            "attestation_hash_match": request.require_attestation_hash_match,
            "recipient_checksum_match": request.require_recipient_checksum_match,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_portfolio_receipt(runs_dir, receipt)
    return receipt


def read_handoff_release_portfolio_receipt(
    runs_dir: Path,
) -> HandoffReleasePortfolioReceipt:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleasePortfolioReceipt, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleasePortfolioReceipt.parse_obj(data)


def write_handoff_release_portfolio_receipt(
    runs_dir: Path,
    receipt: HandoffReleasePortfolioReceipt,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON).write_text(
        json.dumps(_model_to_plain(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD).write_text(
        render_handoff_release_portfolio_receipt_markdown(receipt),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD}",
    ]


def render_handoff_release_portfolio_receipt_markdown(
    receipt: HandoffReleasePortfolioReceipt,
) -> str:
    lines = [
        "# Handoff Release Portfolio Receipt",
        "",
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
        f"- Attestation readiness: `{receipt.attestation_readiness}`",
        f"- Attestation verification readiness: `{receipt.attestation_verification_readiness}`",
        f"- Attestation SHA-256: `{receipt.attestation_sha256 or 'unavailable'}`",
        "- Recipient attestation SHA-256: "
        f"`{receipt.recipient_attestation_sha256 or 'unavailable'}`",
        f"- Releases: {receipt.summary.release_count}",
        f"- Attested artifacts: {receipt.summary.attested_artifacts}",
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
    request: HandoffReleasePortfolioReceiptRequest,
    attestation: HandoffReleaseAttestation,
    verification: HandoffReleaseAttestationVerificationReport,
    current_attestation_sha: str | None,
) -> list[HandoffReleasePortfolioReceiptCheck]:
    return [
        _attestation_ready_check(
            attestation,
            required=request.require_attestation_ready,
        ),
        _attestation_verification_ready_check(
            verification,
            required=request.require_attestation_verification_valid,
        ),
        _attestation_hash_check(
            current_attestation_sha,
            verification,
            required=request.require_attestation_hash_match,
        ),
        _recipient_checksum_check(
            current_attestation_sha,
            request.recipient_attestation_sha256,
            required=request.require_recipient_checksum_match,
        ),
        _transfer_metadata_check(request),
        _recipient_outcome_check(request),
        _operator_audit_check(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]


def _attestation_ready_check(
    attestation: HandoffReleaseAttestation,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptCheck:
    if attestation.readiness != "attested":
        return HandoffReleasePortfolioReceiptCheck(
            check_id="attestation_ready",
            title="Attestation Readiness",
            status="failed" if required else "warning",
            summary=f"Attestation readiness is `{attestation.readiness}`.",
            required_action="Resolve attestation blockers before recording final receipt.",
            evidence_artifacts=attestation.artifacts,
            metadata={"readiness": attestation.readiness, "required": required},
        )
    return HandoffReleasePortfolioReceiptCheck(
        check_id="attestation_ready",
        title="Attestation Readiness",
        status="passed",
        summary="Attestation is ready for portfolio reliance.",
        evidence_artifacts=attestation.artifacts,
        metadata={"readiness": attestation.readiness, "required": required},
    )


def _attestation_verification_ready_check(
    verification: HandoffReleaseAttestationVerificationReport,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptCheck:
    if verification.readiness != "valid":
        return HandoffReleasePortfolioReceiptCheck(
            check_id="attestation_verification",
            title="Attestation Verification",
            status="failed" if required else "warning",
            summary=f"Attestation verification readiness is `{verification.readiness}`.",
            required_action="Resolve attestation verification findings before final receipt.",
            evidence_artifacts=verification.artifacts,
            metadata={"readiness": verification.readiness, "required": required},
        )
    return HandoffReleasePortfolioReceiptCheck(
        check_id="attestation_verification",
        title="Attestation Verification",
        status="passed",
        summary="Attestation verification report is valid.",
        evidence_artifacts=verification.artifacts,
        metadata={"readiness": verification.readiness, "required": required},
    )


def _attestation_hash_check(
    current_attestation_sha: str | None,
    verification: HandoffReleaseAttestationVerificationReport,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptCheck:
    metadata = {
        "expected_attestation_sha256": verification.attestation_sha256,
        "actual_attestation_sha256": current_attestation_sha,
        "required": required,
    }
    if not current_attestation_sha:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="attestation_hash",
            title="Attestation Hash",
            status="failed" if required else "warning",
            summary="Attestation JSON artifact is missing.",
            required_action="Restore or regenerate the attestation before final receipt.",
            evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
            metadata=metadata,
        )
    if current_attestation_sha != verification.attestation_sha256:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="attestation_hash",
            title="Attestation Hash",
            status="failed" if required else "warning",
            summary="Current attestation hash differs from attestation verification.",
            required_action="Regenerate attestation verification or restore the attestation.",
            evidence_artifacts=[
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}",
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}",
            ],
            metadata=metadata,
        )
    return HandoffReleasePortfolioReceiptCheck(
        check_id="attestation_hash",
        title="Attestation Hash",
        status="passed",
        summary="Current attestation hash matches attestation verification.",
        evidence_artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}",
        ],
        metadata=metadata,
    )


def _recipient_checksum_check(
    current_attestation_sha: str | None,
    recipient_hash: str | None,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptCheck:
    normalized = _normalize_optional_hash(recipient_hash)
    metadata = {
        "expected_attestation_sha256": current_attestation_sha,
        "recipient_attestation_sha256": normalized,
        "required": required,
    }
    if not normalized:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient attestation SHA-256 was not provided.",
            required_action="Record the attestation checksum observed by the recipient.",
            evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
            metadata=metadata,
        )
    if not SHA256_PATTERN.match(normalized):
        return HandoffReleasePortfolioReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient attestation SHA-256 is not a valid 64-character hex digest.",
            required_action="Record the exact SHA-256 digest observed by the recipient.",
            evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
            metadata=metadata,
        )
    if not current_attestation_sha:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Local attestation SHA-256 is unavailable for recipient comparison.",
            required_action="Restore or regenerate the attestation before final receipt.",
            evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
            metadata=metadata,
        )
    if current_attestation_sha and normalized != current_attestation_sha.lower():
        return HandoffReleasePortfolioReceiptCheck(
            check_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient attestation SHA-256 does not match the local attestation.",
            required_action="Do not rely on this receipt until the recipient confirms the hash.",
            evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
            metadata=metadata,
        )
    return HandoffReleasePortfolioReceiptCheck(
        check_id="recipient_checksum",
        title="Recipient Checksum",
        status="passed",
        summary="Recipient attestation SHA-256 matches the local attestation.",
        evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
        metadata=metadata,
    )


def _transfer_metadata_check(
    request: HandoffReleasePortfolioReceiptRequest,
) -> HandoffReleasePortfolioReceiptCheck:
    missing = []
    if not request.recipient.strip():
        missing.append("recipient")
    if not request.transfer_method.strip():
        missing.append("transfer_method")
    if not request.transfer_reference.strip():
        missing.append("transfer_reference")
    if missing:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="transfer_metadata",
            title="Transfer Metadata",
            status="warning",
            summary="Portfolio receipt metadata is incomplete: " + ", ".join(missing),
            required_action="Record recipient, transfer method, and transfer reference.",
            metadata={"missing": missing},
        )
    return HandoffReleasePortfolioReceiptCheck(
        check_id="transfer_metadata",
        title="Transfer Metadata",
        status="passed",
        summary="Portfolio receipt metadata is complete.",
        metadata={
            "recipient": request.recipient,
            "transfer_method": request.transfer_method,
            "transfer_reference": request.transfer_reference,
        },
    )


def _recipient_outcome_check(
    request: HandoffReleasePortfolioReceiptRequest,
) -> HandoffReleasePortfolioReceiptCheck:
    if request.outcome == "rejected":
        return HandoffReleasePortfolioReceiptCheck(
            check_id="recipient_outcome",
            title="Recipient Outcome",
            status="failed",
            summary="Recipient rejected the portfolio handoff.",
            required_action="Resolve recipient rejection before closing transfer custody.",
            metadata={"outcome": request.outcome},
        )
    if request.outcome in {"pending", "accepted_with_exceptions"}:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="recipient_outcome",
            title="Recipient Outcome",
            status="warning",
            summary=f"Recipient outcome is `{request.outcome}`.",
            required_action="Track exceptions or pending acceptance to closure.",
            metadata={"outcome": request.outcome},
        )
    return HandoffReleasePortfolioReceiptCheck(
        check_id="recipient_outcome",
        title="Recipient Outcome",
        status="passed",
        summary="Recipient accepted the portfolio handoff.",
        metadata={"outcome": request.outcome},
    )


def _operator_audit_check(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptCheck:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleasePortfolioReceiptCheck(
            check_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before receipt reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleasePortfolioReceiptCheck(
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
    stripped = value.strip().lower()
    return stripped or None


def _file_sha256_or_none(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    return file_sha256(path)


def _ledger_dir(runs_dir: Path) -> Path:
    root = runs_dir.resolve()
    ledger_dir = (root / HANDOFF_LEDGER_DIR).resolve()
    if root != ledger_dir and root not in ledger_dir.parents:
        raise ValueError("Invalid handoff release ledger directory")
    return ledger_dir


def _dedupe(items: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = str(item).strip()
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
