"""Verification reports for handoff release portfolio receipts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release_attestation import (
    HANDOFF_RELEASE_ATTESTATION_JSON,
    read_handoff_release_attestation,
)
from .handoff_release_attestation_verification import (
    HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON,
    read_handoff_release_attestation_verification_report,
)
from .handoff_release_ledger import HANDOFF_LEDGER_DIR
from .handoff_release_portfolio_receipt import (
    HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON,
    HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD,
    HandoffReleasePortfolioReceipt,
    read_handoff_release_portfolio_receipt,
    render_handoff_release_portfolio_receipt_markdown,
)
from .operator_audit import verify_operator_audit

PortfolioReceiptVerificationStatus = Literal["passed", "warning", "failed"]
PortfolioReceiptVerificationReadiness = Literal["valid", "warnings", "failed"]

HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON = (
    "handoff_release_portfolio_receipt_verification.json"
)
HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_MD = (
    "handoff_release_portfolio_receipt_verification.md"
)


class HandoffReleasePortfolioReceiptVerificationRequest(BaseModel):
    requested_by: str = "operator"
    require_receipt_artifacts: bool = True
    require_receipt_recorded: bool = True
    require_attestation_hash_match: bool = True
    require_attestation_verification_valid: bool = True
    require_recipient_checksum_match: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleasePortfolioReceiptVerificationFinding(BaseModel):
    finding_id: str
    title: str
    status: PortfolioReceiptVerificationStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleasePortfolioReceiptVerificationReport(BaseModel):
    report_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    readiness: PortfolioReceiptVerificationReadiness = "warnings"
    receipt_generated_at: str
    receipt_sha256: str | None = None
    current_attestation_sha256: str | None = None
    current_attestation_verification_sha256: str | None = None
    findings: list[HandoffReleasePortfolioReceiptVerificationFinding] = Field(
        default_factory=list
    )
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_portfolio_receipt_verification_report(
    *,
    runs_dir: Path,
    request: HandoffReleasePortfolioReceiptVerificationRequest | None = None,
    receipt: HandoffReleasePortfolioReceipt | None = None,
) -> HandoffReleasePortfolioReceiptVerificationReport:
    request = request or HandoffReleasePortfolioReceiptVerificationRequest()
    receipt = receipt or read_handoff_release_portfolio_receipt(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    ledger_dir = _ledger_dir(runs_dir)
    findings = [
        _receipt_artifacts_finding(
            runs_dir,
            receipt,
            required=request.require_receipt_artifacts,
        ),
        _receipt_readiness_finding(
            receipt,
            required=request.require_receipt_recorded,
        ),
        _attestation_hash_finding(
            runs_dir,
            receipt,
            required=request.require_attestation_hash_match,
        ),
        _attestation_verification_finding(
            runs_dir,
            receipt,
            required=request.require_attestation_verification_valid,
        ),
        _recipient_checksum_finding(
            receipt,
            required=request.require_recipient_checksum_match,
        ),
        _operator_audit_finding(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]
    failures = _dedupe(finding.summary for finding in findings if finding.status == "failed")
    warnings = _dedupe(finding.summary for finding in findings if finding.status == "warning")
    readiness: PortfolioReceiptVerificationReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"

    report = HandoffReleasePortfolioReceiptVerificationReport(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        receipt_generated_at=receipt.generated_at,
        receipt_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON
        ),
        current_attestation_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_ATTESTATION_JSON
        ),
        current_attestation_verification_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON
        ),
        findings=findings,
        failures=failures,
        warnings=warnings,
        required_controls={
            "receipt_artifacts": request.require_receipt_artifacts,
            "receipt_recorded": request.require_receipt_recorded,
            "attestation_hash_match": request.require_attestation_hash_match,
            "attestation_verification_valid": (
                request.require_attestation_verification_valid
            ),
            "recipient_checksum_match": request.require_recipient_checksum_match,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_portfolio_receipt_verification_report(runs_dir, report)
    return report


def read_handoff_release_portfolio_receipt_verification_report(
    runs_dir: Path,
) -> HandoffReleasePortfolioReceiptVerificationReport:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleasePortfolioReceiptVerificationReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleasePortfolioReceiptVerificationReport.parse_obj(data)


def write_handoff_release_portfolio_receipt_verification_report(
    runs_dir: Path,
    report: HandoffReleasePortfolioReceiptVerificationReport,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_MD).write_text(
        render_handoff_release_portfolio_receipt_verification_markdown(report),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_MD}",
    ]


def render_handoff_release_portfolio_receipt_verification_markdown(
    report: HandoffReleasePortfolioReceiptVerificationReport,
) -> str:
    lines = [
        "# Handoff Release Portfolio Receipt Verification",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Receipt generated at: `{report.receipt_generated_at}`",
        f"- Receipt SHA-256: `{report.receipt_sha256 or 'unavailable'}`",
        "- Current attestation SHA-256: "
        f"`{report.current_attestation_sha256 or 'unavailable'}`",
        "- Current attestation verification SHA-256: "
        f"`{report.current_attestation_verification_sha256 or 'unavailable'}`",
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
        lines.extend(f"- {failure}" for failure in report.failures)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    if report.notes:
        lines.extend(["## Notes", "", report.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _receipt_artifacts_finding(
    runs_dir: Path,
    receipt: HandoffReleasePortfolioReceipt,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptVerificationFinding:
    ledger_dir = _ledger_dir(runs_dir)
    expected = [HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON, HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD]
    missing = [name for name in expected if not (ledger_dir / name).exists()]
    if missing:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="receipt_artifacts",
            title="Receipt Artifacts",
            status="failed" if required else "warning",
            summary="Portfolio receipt artifacts are missing: " + ", ".join(missing),
            required_action="Regenerate or restore the portfolio receipt.",
            evidence_artifacts=receipt.artifacts,
            metadata={"missing": missing, "required": required},
        )
    rendered = render_handoff_release_portfolio_receipt_markdown(receipt)
    current_md = (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD).read_text(
        encoding="utf-8"
    )
    if rendered != current_md:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="receipt_artifacts",
            title="Receipt Artifacts",
            status="failed" if required else "warning",
            summary="Receipt Markdown sidecar does not match the JSON receipt content.",
            required_action="Regenerate the portfolio receipt sidecars.",
            evidence_artifacts=receipt.artifacts,
            metadata={"required": required},
        )
    return HandoffReleasePortfolioReceiptVerificationFinding(
        finding_id="receipt_artifacts",
        title="Receipt Artifacts",
        status="passed",
        summary="Receipt JSON and Markdown artifacts are mutually consistent.",
        evidence_artifacts=receipt.artifacts,
        metadata={"required": required},
    )


def _receipt_readiness_finding(
    receipt: HandoffReleasePortfolioReceipt,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptVerificationFinding:
    if receipt.readiness != "recorded":
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="receipt_readiness",
            title="Receipt Readiness",
            status="failed" if required else "warning",
            summary=f"Portfolio receipt readiness is {receipt.readiness}.",
            required_action="Resolve receipt blockers before relying on verification.",
            evidence_artifacts=receipt.artifacts,
            metadata={
                "readiness": receipt.readiness,
                "blockers": receipt.blockers,
                "warnings": receipt.warnings,
                "required": required,
            },
        )
    return HandoffReleasePortfolioReceiptVerificationFinding(
        finding_id="receipt_readiness",
        title="Receipt Readiness",
        status="passed",
        summary="Saved portfolio receipt readiness is recorded.",
        evidence_artifacts=receipt.artifacts,
        metadata={"readiness": receipt.readiness, "required": required},
    )


def _attestation_hash_finding(
    runs_dir: Path,
    receipt: HandoffReleasePortfolioReceipt,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptVerificationFinding:
    ledger_dir = _ledger_dir(runs_dir)
    current_sha = _file_sha256_or_none(ledger_dir / HANDOFF_RELEASE_ATTESTATION_JSON)
    drift: list[str] = []
    if current_sha != receipt.attestation_sha256:
        drift.append("attestation_sha256")
    if receipt.expected_attestation_sha256 and current_sha != receipt.expected_attestation_sha256:
        drift.append("expected_attestation_sha256")
    try:
        attestation = read_handoff_release_attestation(runs_dir)
        if attestation.generated_at != receipt.attestation_generated_at:
            drift.append("attestation_generated_at")
    except FileNotFoundError:
        drift.append("attestation_missing")
    except Exception as e:
        drift.append(f"attestation_unreadable:{type(e).__name__}")
    if drift:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="attestation_hash",
            title="Attestation Hash",
            status="failed" if required else "warning",
            summary="Current attestation differs from the saved portfolio receipt.",
            required_action="Restore the attestation or record a new portfolio receipt.",
            evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
            metadata={
                "drift": drift,
                "receipt_attestation_sha256": receipt.attestation_sha256,
                "receipt_expected_attestation_sha256": receipt.expected_attestation_sha256,
                "current_attestation_sha256": current_sha,
                "required": required,
            },
        )
    return HandoffReleasePortfolioReceiptVerificationFinding(
        finding_id="attestation_hash",
        title="Attestation Hash",
        status="passed",
        summary="Current attestation matches the saved portfolio receipt.",
        evidence_artifacts=[f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}"],
        metadata={"required": required},
    )


def _attestation_verification_finding(
    runs_dir: Path,
    receipt: HandoffReleasePortfolioReceipt,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptVerificationFinding:
    ledger_dir = _ledger_dir(runs_dir)
    verification_sha = _file_sha256_or_none(
        ledger_dir / HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON
    )
    try:
        verification = read_handoff_release_attestation_verification_report(runs_dir)
    except FileNotFoundError:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="attestation_verification",
            title="Attestation Verification",
            status="failed" if required else "warning",
            summary="Attestation verification report is missing.",
            required_action="Regenerate attestation verification before receipt reliance.",
            evidence_artifacts=[
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}"
            ],
            metadata={"required": required},
        )
    except Exception as e:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="attestation_verification",
            title="Attestation Verification",
            status="failed" if required else "warning",
            summary=f"Attestation verification report could not be read: {type(e).__name__}.",
            required_action="Restore or regenerate attestation verification.",
            evidence_artifacts=[
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}"
            ],
            metadata={"error": str(e), "required": required},
        )
    drift: list[str] = []
    if verification.generated_at != receipt.attestation_verification_generated_at:
        drift.append("attestation_verification_generated_at")
    if verification.readiness != receipt.attestation_verification_readiness:
        drift.append("attestation_verification_readiness")
    if verification.attestation_sha256 != receipt.attestation_sha256:
        drift.append("attestation_verification_attestation_sha256")
    if verification.attestation_sha256 != receipt.expected_attestation_sha256:
        drift.append("attestation_verification_expected_attestation_sha256")
    if verification.readiness != "valid":
        drift.append("attestation_verification_not_valid")
    if drift:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="attestation_verification",
            title="Attestation Verification",
            status="failed" if required else "warning",
            summary="Current attestation verification differs from the saved portfolio receipt.",
            required_action="Regenerate attestation verification or record a new receipt.",
            evidence_artifacts=[
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}"
            ],
            metadata={
                "drift": drift,
                "current_attestation_verification_sha256": verification_sha,
                "readiness": verification.readiness,
                "required": required,
            },
        )
    return HandoffReleasePortfolioReceiptVerificationFinding(
        finding_id="attestation_verification",
        title="Attestation Verification",
        status="passed",
        summary="Current attestation verification matches the saved portfolio receipt.",
        evidence_artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}"
        ],
        metadata={
            "current_attestation_verification_sha256": verification_sha,
            "required": required,
        },
    )


def _recipient_checksum_finding(
    receipt: HandoffReleasePortfolioReceipt,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptVerificationFinding:
    expected = receipt.attestation_sha256
    observed = receipt.recipient_attestation_sha256
    if not expected:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Saved receipt does not include a local attestation SHA-256.",
            required_action="Regenerate the portfolio receipt after restoring attestation hash.",
            evidence_artifacts=receipt.artifacts,
            metadata={"recipient_attestation_sha256": observed, "required": required},
        )
    if not observed:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient attestation SHA-256 is missing from the portfolio receipt.",
            required_action="Record a recipient-observed attestation checksum.",
            evidence_artifacts=receipt.artifacts,
            metadata={"expected_attestation_sha256": expected, "required": required},
        )
    if expected and observed != expected.lower():
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="recipient_checksum",
            title="Recipient Checksum",
            status="failed" if required else "warning",
            summary="Recipient attestation SHA-256 differs from the saved receipt hash.",
            required_action="Resolve recipient checksum mismatch before relying on receipt.",
            evidence_artifacts=receipt.artifacts,
            metadata={
                "expected_attestation_sha256": expected,
                "recipient_attestation_sha256": observed,
                "required": required,
            },
        )
    return HandoffReleasePortfolioReceiptVerificationFinding(
        finding_id="recipient_checksum",
        title="Recipient Checksum",
        status="passed",
        summary="Recipient attestation SHA-256 matches the saved receipt hash.",
        evidence_artifacts=receipt.artifacts,
        metadata={
            "expected_attestation_sha256": expected,
            "recipient_attestation_sha256": observed,
            "required": required,
        },
    )


def _operator_audit_finding(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleasePortfolioReceiptVerificationFinding:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleasePortfolioReceiptVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before relying on this report.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleasePortfolioReceiptVerificationFinding(
        finding_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


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


def _dedupe(items: Iterable[str]) -> list[str]:
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
