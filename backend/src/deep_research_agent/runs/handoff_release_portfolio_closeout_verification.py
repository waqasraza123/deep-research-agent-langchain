"""Verification reports for final handoff release portfolio closeouts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release_attestation import read_handoff_release_attestation
from .handoff_release_attestation_verification import (
    read_handoff_release_attestation_verification_report,
)
from .handoff_release_ledger import HANDOFF_LEDGER_DIR, read_handoff_release_ledger
from .handoff_release_ledger_verification import (
    read_handoff_release_ledger_verification_report,
)
from .handoff_release_portfolio_closeout import (
    HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON,
    HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD,
    HandoffReleasePortfolioCloseout,
    HandoffReleasePortfolioCloseoutArtifactHash,
    read_handoff_release_portfolio_closeout,
    render_handoff_release_portfolio_closeout_markdown,
)
from .handoff_release_portfolio_receipt import read_handoff_release_portfolio_receipt
from .handoff_release_portfolio_receipt_verification import (
    read_handoff_release_portfolio_receipt_verification_report,
)
from .operator_audit import verify_operator_audit

CloseoutVerificationStatus = Literal["passed", "warning", "failed"]
CloseoutVerificationReadiness = Literal["valid", "warnings", "failed"]

HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_JSON = (
    "handoff_release_portfolio_closeout_verification.json"
)
HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_MD = (
    "handoff_release_portfolio_closeout_verification.md"
)


class HandoffReleasePortfolioCloseoutVerificationRequest(BaseModel):
    requested_by: str = "operator"
    require_closeout_artifacts: bool = True
    require_closeout_closed: bool = True
    require_final_artifact_hashes: bool = True
    require_upstream_timestamps_match: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleasePortfolioCloseoutVerificationFinding(BaseModel):
    finding_id: str
    title: str
    status: CloseoutVerificationStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleasePortfolioCloseoutArtifactVerification(BaseModel):
    path: str
    status: CloseoutVerificationStatus
    expected_status: str = ""
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    expected_size_bytes: int | None = None
    actual_size_bytes: int | None = None
    summary: str = ""


class HandoffReleasePortfolioCloseoutVerificationReport(BaseModel):
    report_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    readiness: CloseoutVerificationReadiness = "warnings"
    closeout_generated_at: str
    closeout_sha256: str | None = None
    findings: list[HandoffReleasePortfolioCloseoutVerificationFinding] = Field(
        default_factory=list
    )
    artifact_verifications: list[HandoffReleasePortfolioCloseoutArtifactVerification] = Field(
        default_factory=list
    )
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_portfolio_closeout_verification_report(
    *,
    runs_dir: Path,
    request: HandoffReleasePortfolioCloseoutVerificationRequest | None = None,
    closeout: HandoffReleasePortfolioCloseout | None = None,
) -> HandoffReleasePortfolioCloseoutVerificationReport:
    request = request or HandoffReleasePortfolioCloseoutVerificationRequest()
    closeout = closeout or read_handoff_release_portfolio_closeout(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    artifact_verifications = _artifact_verifications(runs_dir, closeout.artifact_hashes)
    findings = [
        _closeout_artifacts_finding(
            runs_dir,
            closeout,
            required=request.require_closeout_artifacts,
        ),
        _closeout_readiness_finding(
            closeout,
            required=request.require_closeout_closed,
        ),
        _artifact_hashes_finding(
            artifact_verifications,
            required=request.require_final_artifact_hashes,
        ),
        _upstream_timestamps_finding(
            runs_dir,
            closeout,
            required=request.require_upstream_timestamps_match,
        ),
        _operator_audit_finding(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]
    failures = [finding.summary for finding in findings if finding.status == "failed"]
    failures.extend(item.summary for item in artifact_verifications if item.status == "failed")
    warnings = [finding.summary for finding in findings if finding.status == "warning"]
    warnings.extend(item.summary for item in artifact_verifications if item.status == "warning")
    failures = _dedupe(failures)
    warnings = _dedupe(warnings)
    readiness: CloseoutVerificationReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"

    ledger_dir = _ledger_dir(runs_dir)
    report = HandoffReleasePortfolioCloseoutVerificationReport(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        closeout_generated_at=closeout.generated_at,
        closeout_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON
        ),
        findings=findings,
        artifact_verifications=artifact_verifications,
        failures=failures,
        warnings=warnings,
        required_controls={
            "closeout_artifacts": request.require_closeout_artifacts,
            "closeout_closed": request.require_closeout_closed,
            "final_artifact_hashes": request.require_final_artifact_hashes,
            "upstream_timestamps_match": request.require_upstream_timestamps_match,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_portfolio_closeout_verification_report(runs_dir, report)
    return report


def read_handoff_release_portfolio_closeout_verification_report(
    runs_dir: Path,
) -> HandoffReleasePortfolioCloseoutVerificationReport:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleasePortfolioCloseoutVerificationReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleasePortfolioCloseoutVerificationReport.parse_obj(data)


def write_handoff_release_portfolio_closeout_verification_report(
    runs_dir: Path,
    report: HandoffReleasePortfolioCloseoutVerificationReport,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_MD).write_text(
        render_handoff_release_portfolio_closeout_verification_markdown(report),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_VERIFICATION_MD}",
    ]


def render_handoff_release_portfolio_closeout_verification_markdown(
    report: HandoffReleasePortfolioCloseoutVerificationReport,
) -> str:
    lines = [
        "# Handoff Release Portfolio Closeout Verification",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Closeout generated at: `{report.closeout_generated_at}`",
        f"- Closeout SHA-256: `{report.closeout_sha256 or 'unavailable'}`",
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
    if report.artifact_verifications:
        lines.extend(["## Artifact Verifications", ""])
        for item in report.artifact_verifications:
            lines.extend(
                [
                    f"### {item.path}",
                    "",
                    f"- Status: `{item.status}`",
                    f"- Expected status: `{item.expected_status}`",
                    f"- Expected SHA-256: `{item.expected_sha256 or 'unavailable'}`",
                    f"- Actual SHA-256: `{item.actual_sha256 or 'unavailable'}`",
                    f"- Expected size bytes: {_display_size(item.expected_size_bytes)}",
                    f"- Actual size bytes: {_display_size(item.actual_size_bytes)}",
                    f"- Summary: {item.summary or 'None'}",
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


def _closeout_artifacts_finding(
    runs_dir: Path,
    closeout: HandoffReleasePortfolioCloseout,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutVerificationFinding:
    ledger_dir = _ledger_dir(runs_dir)
    expected = [HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON, HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD]
    missing = [name for name in expected if not (ledger_dir / name).exists()]
    if missing:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="closeout_artifacts",
            title="Closeout Artifacts",
            status="failed" if required else "warning",
            summary="Portfolio closeout artifacts are missing: " + ", ".join(missing),
            required_action="Regenerate or restore the portfolio closeout.",
            evidence_artifacts=closeout.artifacts,
            metadata={"missing": missing, "required": required},
        )
    rendered = render_handoff_release_portfolio_closeout_markdown(closeout)
    current_md = (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD).read_text(
        encoding="utf-8"
    )
    if rendered != current_md:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="closeout_artifacts",
            title="Closeout Artifacts",
            status="failed" if required else "warning",
            summary="Closeout Markdown sidecar does not match the JSON closeout content.",
            required_action="Regenerate the portfolio closeout sidecars.",
            evidence_artifacts=closeout.artifacts,
            metadata={"required": required},
        )
    return HandoffReleasePortfolioCloseoutVerificationFinding(
        finding_id="closeout_artifacts",
        title="Closeout Artifacts",
        status="passed",
        summary="Closeout JSON and Markdown artifacts are mutually consistent.",
        evidence_artifacts=closeout.artifacts,
        metadata={"required": required},
    )


def _closeout_readiness_finding(
    closeout: HandoffReleasePortfolioCloseout,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutVerificationFinding:
    if closeout.readiness != "closed":
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="closeout_readiness",
            title="Closeout Readiness",
            status="failed" if required else "warning",
            summary=f"Portfolio closeout readiness is {closeout.readiness}.",
            required_action="Resolve closeout blockers before relying on verification.",
            evidence_artifacts=closeout.artifacts,
            metadata={
                "readiness": closeout.readiness,
                "blockers": closeout.blockers,
                "warnings": closeout.warnings,
                "required": required,
            },
        )
    return HandoffReleasePortfolioCloseoutVerificationFinding(
        finding_id="closeout_readiness",
        title="Closeout Readiness",
        status="passed",
        summary="Saved portfolio closeout readiness is closed.",
        evidence_artifacts=closeout.artifacts,
        metadata={"readiness": closeout.readiness, "required": required},
    )


def _artifact_hashes_finding(
    verifications: list[HandoffReleasePortfolioCloseoutArtifactVerification],
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutVerificationFinding:
    failed = [item.path for item in verifications if item.status == "failed"]
    warnings = [item.path for item in verifications if item.status == "warning"]
    if failed:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="final_artifact_hashes",
            title="Final Artifact Hashes",
            status="failed" if required else "warning",
            summary="One or more final closeout artifacts failed hash verification.",
            required_action="Restore changed artifacts or generate a new closeout.",
            evidence_artifacts=[item.path for item in verifications],
            metadata={"failed": failed, "warnings": warnings, "required": required},
        )
    if warnings:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="final_artifact_hashes",
            title="Final Artifact Hashes",
            status="warning",
            summary="One or more final closeout artifacts have verification warnings.",
            required_action="Review artifact warnings before closeout reliance.",
            evidence_artifacts=[item.path for item in verifications],
            metadata={"warnings": warnings, "required": required},
        )
    return HandoffReleasePortfolioCloseoutVerificationFinding(
        finding_id="final_artifact_hashes",
        title="Final Artifact Hashes",
        status="passed",
        summary="Every final closeout artifact hash matches current disk state.",
        evidence_artifacts=[item.path for item in verifications],
        metadata={"artifact_count": len(verifications), "required": required},
    )


def _upstream_timestamps_finding(
    runs_dir: Path,
    closeout: HandoffReleasePortfolioCloseout,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutVerificationFinding:
    drift: list[str] = []
    try:
        ledger = read_handoff_release_ledger(runs_dir)
        if ledger.generated_at != closeout.ledger_generated_at:
            drift.append("ledger_generated_at")
    except Exception as e:
        drift.append(f"ledger_unreadable:{type(e).__name__}")
    try:
        report = read_handoff_release_ledger_verification_report(runs_dir)
        if report.generated_at != closeout.ledger_verification_generated_at:
            drift.append("ledger_verification_generated_at")
    except Exception as e:
        drift.append(f"ledger_verification_unreadable:{type(e).__name__}")
    try:
        attestation = read_handoff_release_attestation(runs_dir)
        if attestation.generated_at != closeout.attestation_generated_at:
            drift.append("attestation_generated_at")
    except Exception as e:
        drift.append(f"attestation_unreadable:{type(e).__name__}")
    try:
        report = read_handoff_release_attestation_verification_report(runs_dir)
        if report.generated_at != closeout.attestation_verification_generated_at:
            drift.append("attestation_verification_generated_at")
    except Exception as e:
        drift.append(f"attestation_verification_unreadable:{type(e).__name__}")
    try:
        receipt = read_handoff_release_portfolio_receipt(runs_dir)
        if receipt.generated_at != closeout.portfolio_receipt_generated_at:
            drift.append("portfolio_receipt_generated_at")
    except Exception as e:
        drift.append(f"portfolio_receipt_unreadable:{type(e).__name__}")
    try:
        report = read_handoff_release_portfolio_receipt_verification_report(runs_dir)
        if report.generated_at != closeout.portfolio_receipt_verification_generated_at:
            drift.append("portfolio_receipt_verification_generated_at")
    except Exception as e:
        drift.append(f"portfolio_receipt_verification_unreadable:{type(e).__name__}")
    if drift:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="upstream_timestamps",
            title="Upstream Timestamps",
            status="failed" if required else "warning",
            summary="One or more upstream custody artifacts differ from the saved closeout.",
            required_action="Regenerate affected controls or generate a new closeout.",
            evidence_artifacts=closeout.artifacts,
            metadata={"drift": drift, "required": required},
        )
    return HandoffReleasePortfolioCloseoutVerificationFinding(
        finding_id="upstream_timestamps",
        title="Upstream Timestamps",
        status="passed",
        summary="Upstream custody artifact timestamps match the saved closeout.",
        evidence_artifacts=closeout.artifacts,
        metadata={"required": required},
    )


def _operator_audit_finding(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutVerificationFinding:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleasePortfolioCloseoutVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before closeout reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleasePortfolioCloseoutVerificationFinding(
        finding_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _artifact_verifications(
    runs_dir: Path,
    artifact_hashes: list[HandoffReleasePortfolioCloseoutArtifactHash],
) -> list[HandoffReleasePortfolioCloseoutArtifactVerification]:
    return [_artifact_verification(runs_dir, artifact) for artifact in artifact_hashes]


def _artifact_verification(
    runs_dir: Path,
    artifact: HandoffReleasePortfolioCloseoutArtifactHash,
) -> HandoffReleasePortfolioCloseoutArtifactVerification:
    root = runs_dir.resolve()
    path = (root / artifact.path).resolve()
    if root != path and root not in path.parents:
        return HandoffReleasePortfolioCloseoutArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            expected_size_bytes=artifact.size_bytes,
            summary="Artifact path resolves outside runs_dir.",
        )
    if not path.exists() or path.is_dir():
        return HandoffReleasePortfolioCloseoutArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            expected_size_bytes=artifact.size_bytes,
            summary="Artifact is missing.",
        )
    actual_sha256 = file_sha256(path)
    actual_size = path.stat().st_size
    if artifact.status != "hashed":
        return HandoffReleasePortfolioCloseoutArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            actual_sha256=actual_sha256,
            expected_size_bytes=artifact.size_bytes,
            actual_size_bytes=actual_size,
            summary="Artifact exists now but was not hashed in the saved closeout.",
        )
    if actual_sha256 != artifact.sha256 or actual_size != artifact.size_bytes:
        return HandoffReleasePortfolioCloseoutArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            actual_sha256=actual_sha256,
            expected_size_bytes=artifact.size_bytes,
            actual_size_bytes=actual_size,
            summary="Artifact hash or size differs from the saved closeout.",
        )
    return HandoffReleasePortfolioCloseoutArtifactVerification(
        path=artifact.path,
        status="passed",
        expected_status=artifact.status,
        expected_sha256=artifact.sha256,
        actual_sha256=actual_sha256,
        expected_size_bytes=artifact.size_bytes,
        actual_size_bytes=actual_size,
        summary="Artifact matches the saved closeout.",
    )


def _file_sha256_or_none(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    return file_sha256(path)


def _display_size(size: int | None) -> str:
    return str(size) if size is not None else "unknown"


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
