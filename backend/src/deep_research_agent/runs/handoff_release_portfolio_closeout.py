"""Final closeout manifests for handoff release transfer portfolios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release_attestation import (
    HANDOFF_RELEASE_ATTESTATION_JSON,
    HANDOFF_RELEASE_ATTESTATION_MD,
    read_handoff_release_attestation,
)
from .handoff_release_attestation_verification import (
    HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON,
    HANDOFF_RELEASE_ATTESTATION_VERIFICATION_MD,
    read_handoff_release_attestation_verification_report,
)
from .handoff_release_ledger import (
    HANDOFF_LEDGER_DIR,
    HANDOFF_RELEASE_LEDGER_JSON,
    HANDOFF_RELEASE_LEDGER_MD,
    read_handoff_release_ledger,
)
from .handoff_release_ledger_verification import (
    HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON,
    HANDOFF_RELEASE_LEDGER_VERIFICATION_MD,
    read_handoff_release_ledger_verification_report,
)
from .handoff_release_portfolio_receipt import (
    HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON,
    HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD,
    read_handoff_release_portfolio_receipt,
)
from .handoff_release_portfolio_receipt_verification import (
    HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON,
    HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_MD,
    read_handoff_release_portfolio_receipt_verification_report,
)
from .operator_audit import (
    GLOBAL_AUDIT_DIR,
    OPERATOR_AUDIT_JSONL,
    OPERATOR_AUDIT_MD,
    verify_operator_audit,
)

CloseoutCheckStatus = Literal["passed", "warning", "failed"]
CloseoutArtifactStatus = Literal["hashed", "missing", "unsafe"]
CloseoutReadiness = Literal["closed", "warnings", "blocked"]

HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON = "handoff_release_portfolio_closeout.json"
HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD = "handoff_release_portfolio_closeout.md"


class HandoffReleasePortfolioCloseoutRequest(BaseModel):
    requested_by: str = "operator"
    closeout_scope: str = "release_transfer_portfolio"
    require_ledger_complete: bool = True
    require_ledger_verification_valid: bool = True
    require_attestation_ready: bool = True
    require_attestation_verification_valid: bool = True
    require_portfolio_receipt_recorded: bool = True
    require_portfolio_receipt_verification_valid: bool = True
    require_final_artifact_presence: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleasePortfolioCloseoutCheck(BaseModel):
    check_id: str
    title: str
    status: CloseoutCheckStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleasePortfolioCloseoutArtifactHash(BaseModel):
    path: str
    status: CloseoutArtifactStatus = "missing"
    sha256: str | None = None
    size_bytes: int | None = None
    summary: str = ""


class HandoffReleasePortfolioCloseoutSummary(BaseModel):
    release_count: int = 0
    attested_artifacts: int = 0
    final_artifact_count: int = 0
    missing_artifact_count: int = 0
    unsafe_artifact_count: int = 0
    global_operator_audit_valid: bool = True
    global_operator_audit_event_count: int = 0
    checks_passed: int = 0
    checks_warning: int = 0
    checks_failed: int = 0


class HandoffReleasePortfolioCloseout(BaseModel):
    closeout_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    closeout_scope: str = "release_transfer_portfolio"
    readiness: CloseoutReadiness = "warnings"
    ledger_generated_at: str = ""
    ledger_verification_generated_at: str = ""
    attestation_generated_at: str = ""
    attestation_verification_generated_at: str = ""
    portfolio_receipt_generated_at: str = ""
    portfolio_receipt_verification_generated_at: str = ""
    summary: HandoffReleasePortfolioCloseoutSummary
    checks: list[HandoffReleasePortfolioCloseoutCheck] = Field(default_factory=list)
    artifact_hashes: list[HandoffReleasePortfolioCloseoutArtifactHash] = Field(
        default_factory=list
    )
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_portfolio_closeout(
    *,
    runs_dir: Path,
    request: HandoffReleasePortfolioCloseoutRequest | None = None,
) -> HandoffReleasePortfolioCloseout:
    request = request or HandoffReleasePortfolioCloseoutRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    scope = request.closeout_scope.strip() or "release_transfer_portfolio"

    ledger = read_handoff_release_ledger(runs_dir)
    ledger_verification = read_handoff_release_ledger_verification_report(runs_dir)
    attestation = read_handoff_release_attestation(runs_dir)
    attestation_verification = read_handoff_release_attestation_verification_report(runs_dir)
    receipt = read_handoff_release_portfolio_receipt(runs_dir)
    receipt_verification = read_handoff_release_portfolio_receipt_verification_report(runs_dir)

    artifact_hashes = _final_artifact_hashes(runs_dir)
    checks = [
        _ledger_check(ledger, required=request.require_ledger_complete),
        _ledger_verification_check(
            ledger_verification,
            required=request.require_ledger_verification_valid,
        ),
        _attestation_check(
            attestation,
            required=request.require_attestation_ready,
        ),
        _attestation_verification_check(
            attestation_verification,
            required=request.require_attestation_verification_valid,
        ),
        _portfolio_receipt_check(
            receipt,
            required=request.require_portfolio_receipt_recorded,
        ),
        _portfolio_receipt_verification_check(
            receipt_verification,
            required=request.require_portfolio_receipt_verification_valid,
        ),
        _cross_artifact_consistency_check(
            attestation,
            attestation_verification,
            receipt,
            receipt_verification,
        ),
        _final_artifact_presence_check(
            artifact_hashes,
            required=request.require_final_artifact_presence,
        ),
        _operator_audit_check(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]
    blockers = _dedupe(check.summary for check in checks if check.status == "failed")
    warnings = _dedupe(check.summary for check in checks if check.status == "warning")
    readiness: CloseoutReadiness = "closed"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "warnings"

    audit_valid, audit_events = _operator_audit_summary(runs_dir)
    closeout = HandoffReleasePortfolioCloseout(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        closeout_scope=scope,
        readiness=readiness,
        ledger_generated_at=ledger.generated_at,
        ledger_verification_generated_at=ledger_verification.generated_at,
        attestation_generated_at=attestation.generated_at,
        attestation_verification_generated_at=attestation_verification.generated_at,
        portfolio_receipt_generated_at=receipt.generated_at,
        portfolio_receipt_verification_generated_at=receipt_verification.generated_at,
        summary=HandoffReleasePortfolioCloseoutSummary(
            release_count=attestation.summary.release_count,
            attested_artifacts=attestation.summary.artifact_count,
            final_artifact_count=len(artifact_hashes),
            missing_artifact_count=sum(1 for item in artifact_hashes if item.status == "missing"),
            unsafe_artifact_count=sum(1 for item in artifact_hashes if item.status == "unsafe"),
            global_operator_audit_valid=audit_valid,
            global_operator_audit_event_count=audit_events,
            checks_passed=sum(1 for check in checks if check.status == "passed"),
            checks_warning=sum(1 for check in checks if check.status == "warning"),
            checks_failed=sum(1 for check in checks if check.status == "failed"),
        ),
        checks=checks,
        artifact_hashes=artifact_hashes,
        blockers=blockers,
        warnings=warnings,
        required_controls={
            "ledger_complete": request.require_ledger_complete,
            "ledger_verification_valid": request.require_ledger_verification_valid,
            "attestation_ready": request.require_attestation_ready,
            "attestation_verification_valid": request.require_attestation_verification_valid,
            "portfolio_receipt_recorded": request.require_portfolio_receipt_recorded,
            "portfolio_receipt_verification_valid": (
                request.require_portfolio_receipt_verification_valid
            ),
            "final_artifact_presence": request.require_final_artifact_presence,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_portfolio_closeout(runs_dir, closeout)
    return closeout


def read_handoff_release_portfolio_closeout(
    runs_dir: Path,
) -> HandoffReleasePortfolioCloseout:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleasePortfolioCloseout, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleasePortfolioCloseout.parse_obj(data)


def write_handoff_release_portfolio_closeout(
    runs_dir: Path,
    closeout: HandoffReleasePortfolioCloseout,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON).write_text(
        json.dumps(_model_to_plain(closeout), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD).write_text(
        render_handoff_release_portfolio_closeout_markdown(closeout),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_CLOSEOUT_MD}",
    ]


def render_handoff_release_portfolio_closeout_markdown(
    closeout: HandoffReleasePortfolioCloseout,
) -> str:
    lines = [
        "# Handoff Release Portfolio Closeout",
        "",
        f"- Generated at: `{closeout.generated_at}`",
        f"- Requested by: `{closeout.requested_by}`",
        f"- Scope: `{closeout.closeout_scope}`",
        f"- Readiness: `{closeout.readiness}`",
        f"- Releases: {closeout.summary.release_count}",
        f"- Attested artifacts: {closeout.summary.attested_artifacts}",
        f"- Final artifacts: {closeout.summary.final_artifact_count}",
        f"- Missing artifacts: {closeout.summary.missing_artifact_count}",
        f"- Unsafe artifacts: {closeout.summary.unsafe_artifact_count}",
        f"- Global audit valid: `{closeout.summary.global_operator_audit_valid}`",
        f"- Global audit events: {closeout.summary.global_operator_audit_event_count}",
        "",
        "## Checks",
        "",
    ]
    for check in closeout.checks:
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
    if closeout.artifact_hashes:
        lines.extend(["## Final Artifact Hashes", ""])
        for item in closeout.artifact_hashes:
            lines.extend(
                [
                    f"### {item.path}",
                    "",
                    f"- Status: `{item.status}`",
                    f"- SHA-256: `{item.sha256 or 'unavailable'}`",
                    f"- Size bytes: {_display_size(item.size_bytes)}",
                    f"- Summary: {item.summary or 'None'}",
                    "",
                ]
            )
    if closeout.blockers:
        lines.extend(["## Closeout Blockers", ""])
        lines.extend(f"- {item}" for item in closeout.blockers)
        lines.append("")
    if closeout.warnings:
        lines.extend(["## Closeout Warnings", ""])
        lines.extend(f"- {item}" for item in closeout.warnings)
        lines.append("")
    if closeout.notes:
        lines.extend(["## Notes", "", closeout.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _ledger_check(ledger: Any, *, required: bool) -> HandoffReleasePortfolioCloseoutCheck:
    if ledger.summary.indexed_releases < 1:
        return _check(
            "ledger",
            "Release Ledger",
            "failed" if required else "warning",
            "Ledger does not contain any indexed releases.",
            "Generate releases before closing portfolio custody.",
            ledger.artifacts,
            {"required": required},
        )
    if ledger.summary.blocked or ledger.summary.needs_attention:
        return _check(
            "ledger",
            "Release Ledger",
            "failed" if required else "warning",
            "Ledger contains blocked or needs-attention releases.",
            "Resolve ledger custody issues before closeout.",
            ledger.artifacts,
            {
                "blocked": ledger.summary.blocked,
                "needs_attention": ledger.summary.needs_attention,
                "required": required,
            },
        )
    return _check(
        "ledger",
        "Release Ledger",
        "passed",
        "Release ledger is complete.",
        "",
        ledger.artifacts,
        {"required": required},
    )


def _ledger_verification_check(
    report: Any,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    return _readiness_check(
        check_id="ledger_verification",
        title="Ledger Verification",
        readiness=report.readiness,
        expected="valid",
        required=required,
        evidence=report.artifacts,
        required_action="Resolve ledger verification before closeout.",
    )


def _attestation_check(
    attestation: Any,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    return _readiness_check(
        check_id="attestation",
        title="Portfolio Attestation",
        readiness=attestation.readiness,
        expected="attested",
        required=required,
        evidence=attestation.artifacts,
        required_action="Resolve attestation blockers before closeout.",
    )


def _attestation_verification_check(
    report: Any,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    return _readiness_check(
        check_id="attestation_verification",
        title="Attestation Verification",
        readiness=report.readiness,
        expected="valid",
        required=required,
        evidence=report.artifacts,
        required_action="Resolve attestation verification before closeout.",
    )


def _portfolio_receipt_check(
    receipt: Any,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    return _readiness_check(
        check_id="portfolio_receipt",
        title="Portfolio Receipt",
        readiness=receipt.readiness,
        expected="recorded",
        required=required,
        evidence=receipt.artifacts,
        required_action="Resolve portfolio receipt blockers before closeout.",
    )


def _portfolio_receipt_verification_check(
    report: Any,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    return _readiness_check(
        check_id="portfolio_receipt_verification",
        title="Portfolio Receipt Verification",
        readiness=report.readiness,
        expected="valid",
        required=required,
        evidence=report.artifacts,
        required_action="Resolve portfolio receipt verification before closeout.",
    )


def _cross_artifact_consistency_check(
    attestation: Any,
    attestation_verification: Any,
    receipt: Any,
    receipt_verification: Any,
) -> HandoffReleasePortfolioCloseoutCheck:
    drift = []
    if attestation_verification.attestation_sha256 != receipt.attestation_sha256:
        drift.append("attestation_verification_to_receipt")
    if receipt_verification.current_attestation_sha256 != receipt.attestation_sha256:
        drift.append("receipt_verification_to_receipt")
    if receipt.expected_attestation_sha256 != receipt.attestation_sha256:
        drift.append("receipt_expected_attestation_sha256")
    if attestation.generated_at != receipt.attestation_generated_at:
        drift.append("attestation_generated_at")
    if attestation_verification.generated_at != receipt.attestation_verification_generated_at:
        drift.append("attestation_verification_generated_at")
    if drift:
        return _check(
            "cross_artifact_consistency",
            "Cross-Artifact Consistency",
            "failed",
            "Final portfolio custody artifacts disagree on attestation identity.",
            "Regenerate affected controls and record a new closeout.",
            _core_artifact_paths(),
            {"drift": drift},
        )
    return _check(
        "cross_artifact_consistency",
        "Cross-Artifact Consistency",
        "passed",
        "Final portfolio custody artifacts agree on attestation identity.",
        "",
        _core_artifact_paths(),
        {},
    )


def _final_artifact_presence_check(
    hashes: list[HandoffReleasePortfolioCloseoutArtifactHash],
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    missing = [item.path for item in hashes if item.status == "missing"]
    unsafe = [item.path for item in hashes if item.status == "unsafe"]
    if missing or unsafe:
        return _check(
            "final_artifact_presence",
            "Final Artifact Presence",
            "failed" if required else "warning",
            f"{len(missing)} artifacts are missing and {len(unsafe)} paths are unsafe.",
            "Restore missing final artifacts before closeout reliance.",
            [item.path for item in hashes],
            {"missing": missing, "unsafe": unsafe, "required": required},
        )
    return _check(
        "final_artifact_presence",
        "Final Artifact Presence",
        "passed",
        "Every final closeout artifact is present and hashed.",
        "",
        [item.path for item in hashes],
        {"artifact_count": len(hashes), "required": required},
    )


def _operator_audit_check(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleasePortfolioCloseoutCheck:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return _check(
            "global_operator_audit",
            "Global Operator Audit",
            "failed" if required else "warning",
            f"Global operator audit could not be verified: {e}",
            "Repair or investigate the global operator audit log.",
            ["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            {"required": required},
        )
    if not verification.valid:
        return _check(
            "global_operator_audit",
            "Global Operator Audit",
            "failed" if required else "warning",
            "Global operator audit hash-chain verification failed.",
            "Investigate audit log corruption before closeout reliance.",
            ["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            _model_to_plain(verification) | {"required": required},
        )
    return _check(
        "global_operator_audit",
        "Global Operator Audit",
        "passed",
        "Global operator audit hash chain verifies.",
        "",
        ["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        _model_to_plain(verification) | {"required": required},
    )


def _readiness_check(
    *,
    check_id: str,
    title: str,
    readiness: str,
    expected: str,
    required: bool,
    evidence: list[str],
    required_action: str,
) -> HandoffReleasePortfolioCloseoutCheck:
    if readiness != expected:
        return _check(
            check_id,
            title,
            "failed" if required else "warning",
            f"{title} readiness is `{readiness}`.",
            required_action,
            evidence,
            {"readiness": readiness, "expected": expected, "required": required},
        )
    return _check(
        check_id,
        title,
        "passed",
        f"{title} readiness is `{expected}`.",
        "",
        evidence,
        {"readiness": readiness, "expected": expected, "required": required},
    )


def _check(
    check_id: str,
    title: str,
    status: CloseoutCheckStatus,
    summary: str,
    required_action: str,
    evidence: list[str],
    metadata: dict[str, Any],
) -> HandoffReleasePortfolioCloseoutCheck:
    return HandoffReleasePortfolioCloseoutCheck(
        check_id=check_id,
        title=title,
        status=status,
        summary=summary,
        required_action=required_action,
        evidence_artifacts=evidence,
        metadata=metadata,
    )


def _final_artifact_hashes(runs_dir: Path) -> list[HandoffReleasePortfolioCloseoutArtifactHash]:
    return [_artifact_hash(runs_dir, path) for path in _core_artifact_paths()]


def _core_artifact_paths() -> list[str]:
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_MD}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_MD}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_MD}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_MD}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_MD}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_PORTFOLIO_RECEIPT_VERIFICATION_MD}",
        f"{GLOBAL_AUDIT_DIR}/{OPERATOR_AUDIT_JSONL}",
        f"{GLOBAL_AUDIT_DIR}/{OPERATOR_AUDIT_MD}",
    ]


def _artifact_hash(
    runs_dir: Path,
    rel_path: str,
) -> HandoffReleasePortfolioCloseoutArtifactHash:
    root = runs_dir.resolve()
    path = (root / rel_path).resolve()
    if root != path and root not in path.parents:
        return HandoffReleasePortfolioCloseoutArtifactHash(
            path=rel_path,
            status="unsafe",
            summary="Artifact path resolves outside runs_dir.",
        )
    if not path.exists() or path.is_dir():
        return HandoffReleasePortfolioCloseoutArtifactHash(
            path=rel_path,
            status="missing",
            summary="Artifact is missing.",
        )
    return HandoffReleasePortfolioCloseoutArtifactHash(
        path=rel_path,
        status="hashed",
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        summary="Artifact hash recorded.",
    )


def _operator_audit_summary(runs_dir: Path) -> tuple[bool, int]:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError:
        return False, 0
    return verification.valid, verification.event_count


def _display_size(size: int | None) -> str:
    return str(size) if size is not None else "unknown"


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
