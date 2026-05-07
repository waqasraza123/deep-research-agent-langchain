"""Portfolio-level attestations for handoff release custody artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_release_ledger import (
    HANDOFF_LEDGER_DIR,
    HANDOFF_RELEASE_LEDGER_JSON,
    HANDOFF_RELEASE_LEDGER_MD,
    HandoffReleaseLedger,
    read_handoff_release_ledger,
)
from .handoff_release_ledger_verification import (
    HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON,
    HANDOFF_RELEASE_LEDGER_VERIFICATION_MD,
    HandoffReleaseLedgerVerificationReport,
    read_handoff_release_ledger_verification_report,
)
from .operator_audit import verify_operator_audit

AttestationCheckStatus = Literal["passed", "warning", "failed"]
AttestationArtifactStatus = Literal["hashed", "missing", "unsafe"]
AttestationReadiness = Literal["attested", "warnings", "blocked"]

HANDOFF_RELEASE_ATTESTATION_JSON = "handoff_release_attestation.json"
HANDOFF_RELEASE_ATTESTATION_MD = "handoff_release_attestation.md"


class HandoffReleaseAttestationRequest(BaseModel):
    requested_by: str = "operator"
    attestation_scope: str = "release_transfer_portfolio"
    require_ledger_complete: bool = True
    require_ledger_verification_valid: bool = True
    require_global_operator_audit: bool = True
    require_artifact_presence: bool = True
    include_release_artifact_hashes: bool = True
    max_release_artifacts: int = Field(default=5000, ge=1, le=50000)
    notes: str = ""


class HandoffReleaseAttestationCheck(BaseModel):
    check_id: str
    title: str
    status: AttestationCheckStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleaseAttestationArtifactHash(BaseModel):
    path: str
    kind: str = "release_artifact"
    status: AttestationArtifactStatus = "missing"
    sha256: str | None = None
    size_bytes: int | None = None
    summary: str = ""


class HandoffReleaseAttestationSummary(BaseModel):
    release_count: int = 0
    ledger_complete: int = 0
    ledger_needs_attention: int = 0
    ledger_blocked: int = 0
    ledger_verification_readiness: str = "missing"
    artifact_count: int = 0
    missing_artifact_count: int = 0
    unsafe_artifact_count: int = 0
    omitted_release_artifact_count: int = 0
    global_operator_audit_valid: bool = True
    global_operator_audit_event_count: int = 0


class HandoffReleaseAttestation(BaseModel):
    attestation_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    attestation_scope: str = "release_transfer_portfolio"
    readiness: AttestationReadiness = "warnings"
    ledger_generated_at: str
    ledger_verification_generated_at: str
    ledger_sha256: str | None = None
    ledger_verification_sha256: str | None = None
    summary: HandoffReleaseAttestationSummary
    checks: list[HandoffReleaseAttestationCheck] = Field(default_factory=list)
    artifact_hashes: list[HandoffReleaseAttestationArtifactHash] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool | int] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_attestation(
    *,
    runs_dir: Path,
    request: HandoffReleaseAttestationRequest | None = None,
    ledger: HandoffReleaseLedger | None = None,
    verification: HandoffReleaseLedgerVerificationReport | None = None,
) -> HandoffReleaseAttestation:
    request = request or HandoffReleaseAttestationRequest()
    ledger = ledger or read_handoff_release_ledger(runs_dir)
    verification = verification or read_handoff_release_ledger_verification_report(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    scope = (
        request.attestation_scope.strip()
        if request.attestation_scope.strip()
        else "release_transfer_portfolio"
    )

    artifact_hashes, artifact_warning = _artifact_hashes(runs_dir, ledger, verification, request)
    checks = [
        _ledger_completeness_check(ledger, required=request.require_ledger_complete),
        _ledger_verification_check(
            verification,
            required=request.require_ledger_verification_valid,
        ),
        _operator_audit_check(runs_dir, required=request.require_global_operator_audit),
        _artifact_inventory_check(
            artifact_hashes,
            artifact_warning=artifact_warning,
            required=request.require_artifact_presence,
        ),
    ]
    blockers = _dedupe(check.summary for check in checks if check.status == "failed")
    warnings = _dedupe(check.summary for check in checks if check.status == "warning")
    readiness: AttestationReadiness = "attested"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "warnings"

    audit_valid, audit_event_count = _operator_audit_summary(runs_dir)
    summary = HandoffReleaseAttestationSummary(
        release_count=ledger.summary.indexed_releases,
        ledger_complete=ledger.summary.complete,
        ledger_needs_attention=ledger.summary.needs_attention,
        ledger_blocked=ledger.summary.blocked,
        ledger_verification_readiness=verification.readiness,
        artifact_count=len(artifact_hashes),
        missing_artifact_count=sum(1 for item in artifact_hashes if item.status == "missing"),
        unsafe_artifact_count=sum(1 for item in artifact_hashes if item.status == "unsafe"),
        omitted_release_artifact_count=_omitted_release_artifact_count(
            ledger,
            request.max_release_artifacts,
            include_release_artifacts=request.include_release_artifact_hashes,
        ),
        global_operator_audit_valid=audit_valid,
        global_operator_audit_event_count=audit_event_count,
    )
    ledger_dir = _ledger_dir(runs_dir)
    attestation = HandoffReleaseAttestation(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        attestation_scope=scope,
        readiness=readiness,
        ledger_generated_at=ledger.generated_at,
        ledger_verification_generated_at=verification.generated_at,
        ledger_sha256=_file_sha256_or_none(ledger_dir / HANDOFF_RELEASE_LEDGER_JSON),
        ledger_verification_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON
        ),
        summary=summary,
        checks=checks,
        artifact_hashes=artifact_hashes,
        blockers=blockers,
        warnings=warnings,
        required_controls={
            "ledger_complete": request.require_ledger_complete,
            "ledger_verification_valid": request.require_ledger_verification_valid,
            "global_operator_audit": request.require_global_operator_audit,
            "artifact_presence": request.require_artifact_presence,
            "include_release_artifact_hashes": request.include_release_artifact_hashes,
            "max_release_artifacts": request.max_release_artifacts,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_attestation(runs_dir, attestation)
    return attestation


def read_handoff_release_attestation(runs_dir: Path) -> HandoffReleaseAttestation:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_ATTESTATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseAttestation, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseAttestation.parse_obj(data)


def write_handoff_release_attestation(
    runs_dir: Path,
    attestation: HandoffReleaseAttestation,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_ATTESTATION_JSON).write_text(
        json.dumps(
            _model_to_plain(attestation),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_ATTESTATION_MD).write_text(
        render_handoff_release_attestation_markdown(attestation),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_MD}",
    ]


def render_handoff_release_attestation_markdown(
    attestation: HandoffReleaseAttestation,
) -> str:
    lines = [
        "# Handoff Release Attestation",
        "",
        f"- Generated at: `{attestation.generated_at}`",
        f"- Requested by: `{attestation.requested_by}`",
        f"- Scope: `{attestation.attestation_scope}`",
        f"- Readiness: `{attestation.readiness}`",
        f"- Ledger generated at: `{attestation.ledger_generated_at}`",
        f"- Ledger verification generated at: `{attestation.ledger_verification_generated_at}`",
        f"- Ledger SHA-256: `{attestation.ledger_sha256 or 'unavailable'}`",
        "- Ledger verification SHA-256: "
        f"`{attestation.ledger_verification_sha256 or 'unavailable'}`",
        f"- Releases: {attestation.summary.release_count}",
        f"- Hashed artifacts: {attestation.summary.artifact_count}",
        f"- Missing artifacts: {attestation.summary.missing_artifact_count}",
        f"- Unsafe artifact references: {attestation.summary.unsafe_artifact_count}",
        f"- Omitted release artifacts: {attestation.summary.omitted_release_artifact_count}",
        "",
        "## Checks",
        "",
    ]
    for check in attestation.checks:
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
    if attestation.artifact_hashes:
        lines.extend(["## Artifact Hashes", ""])
        for item in attestation.artifact_hashes:
            lines.extend(
                [
                    f"### {item.path}",
                    "",
                    f"- Kind: `{item.kind}`",
                    f"- Status: `{item.status}`",
                    f"- SHA-256: `{item.sha256 or 'unavailable'}`",
                    "- Size bytes: "
                    f"{item.size_bytes if item.size_bytes is not None else 'unknown'}",
                    f"- Summary: {item.summary or 'None'}",
                    "",
                ]
            )
    if attestation.blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in attestation.blockers)
        lines.append("")
    if attestation.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in attestation.warnings)
        lines.append("")
    if attestation.notes:
        lines.extend(["## Notes", "", attestation.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _ledger_completeness_check(
    ledger: HandoffReleaseLedger,
    *,
    required: bool,
) -> HandoffReleaseAttestationCheck:
    if ledger.summary.indexed_releases < 1:
        return HandoffReleaseAttestationCheck(
            check_id="ledger_completeness",
            title="Ledger Completeness",
            status="failed" if required else "warning",
            summary="Ledger does not contain any indexed releases.",
            required_action="Generate at least one release before portfolio attestation.",
            evidence_artifacts=ledger.artifacts,
            metadata=_model_to_plain(ledger.summary) | {"required": required},
        )
    incomplete = ledger.summary.blocked + ledger.summary.needs_attention
    if incomplete:
        return HandoffReleaseAttestationCheck(
            check_id="ledger_completeness",
            title="Ledger Completeness",
            status="failed" if required else "warning",
            summary=(
                "Ledger contains "
                f"{ledger.summary.blocked} blocked and "
                f"{ledger.summary.needs_attention} needs-attention releases."
            ),
            required_action="Resolve custody blockers and regenerate the release ledger.",
            evidence_artifacts=ledger.artifacts,
            metadata=_model_to_plain(ledger.summary) | {"required": required},
        )
    return HandoffReleaseAttestationCheck(
        check_id="ledger_completeness",
        title="Ledger Completeness",
        status="passed",
        summary="Every indexed release is complete in the saved custody ledger.",
        evidence_artifacts=ledger.artifacts,
        metadata=_model_to_plain(ledger.summary) | {"required": required},
    )


def _ledger_verification_check(
    verification: HandoffReleaseLedgerVerificationReport,
    *,
    required: bool,
) -> HandoffReleaseAttestationCheck:
    if verification.readiness != "valid":
        return HandoffReleaseAttestationCheck(
            check_id="ledger_verification",
            title="Ledger Verification",
            status="failed" if required else "warning",
            summary=f"Ledger verification readiness is {verification.readiness}.",
            required_action="Resolve ledger verification findings before relying on attestation.",
            evidence_artifacts=verification.artifacts,
            metadata={
                "readiness": verification.readiness,
                "failures": verification.failures,
                "warnings": verification.warnings,
                "required": required,
            },
        )
    return HandoffReleaseAttestationCheck(
        check_id="ledger_verification",
        title="Ledger Verification",
        status="passed",
        summary="Saved ledger verification is valid.",
        evidence_artifacts=verification.artifacts,
        metadata={
            "readiness": verification.readiness,
            "ledger_sha256": verification.ledger_sha256,
            "required": required,
        },
    )


def _operator_audit_check(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleaseAttestationCheck:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleaseAttestationCheck(
            check_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleaseAttestationCheck(
            check_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before attestation reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleaseAttestationCheck(
        check_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _operator_audit_summary(runs_dir: Path) -> tuple[bool, int]:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError:
        return False, 0
    return verification.valid, verification.event_count


def _artifact_inventory_check(
    artifact_hashes: list[HandoffReleaseAttestationArtifactHash],
    *,
    artifact_warning: str,
    required: bool,
) -> HandoffReleaseAttestationCheck:
    missing = [item.path for item in artifact_hashes if item.status == "missing"]
    unsafe = [item.path for item in artifact_hashes if item.status == "unsafe"]
    if missing or unsafe:
        return HandoffReleaseAttestationCheck(
            check_id="artifact_inventory",
            title="Artifact Inventory",
            status="failed" if required else "warning",
            summary=(
                f"{len(missing)} artifacts are missing and "
                f"{len(unsafe)} artifact references are unsafe."
            ),
            required_action="Restore missing artifacts and remove unsafe references.",
            evidence_artifacts=[item.path for item in artifact_hashes],
            metadata={
                "missing": missing,
                "unsafe": unsafe,
                "artifact_warning": artifact_warning,
                "required": required,
            },
        )
    if artifact_warning:
        return HandoffReleaseAttestationCheck(
            check_id="artifact_inventory",
            title="Artifact Inventory",
            status="warning",
            summary=artifact_warning,
            required_action="Raise max_release_artifacts or narrow the release portfolio.",
            evidence_artifacts=[item.path for item in artifact_hashes],
            metadata={"required": required},
        )
    return HandoffReleaseAttestationCheck(
        check_id="artifact_inventory",
        title="Artifact Inventory",
        status="passed",
        summary="Every attested artifact was hashed successfully.",
        evidence_artifacts=[item.path for item in artifact_hashes],
        metadata={"artifact_count": len(artifact_hashes), "required": required},
    )


def _artifact_hashes(
    runs_dir: Path,
    ledger: HandoffReleaseLedger,
    verification: HandoffReleaseLedgerVerificationReport,
    request: HandoffReleaseAttestationRequest,
) -> tuple[list[HandoffReleaseAttestationArtifactHash], str]:
    base_artifacts = [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_MD}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_MD}",
    ]
    release_artifacts: list[str] = []
    if request.include_release_artifact_hashes:
        for item in ledger.items:
            release_artifacts.extend(item.artifacts)
    release_artifacts = _dedupe(release_artifacts)
    artifact_warning = ""
    if len(release_artifacts) > request.max_release_artifacts:
        artifact_warning = (
            "Attestation hashed "
            f"{request.max_release_artifacts} of {len(release_artifacts)} release artifacts "
            "because max_release_artifacts was reached."
        )
        release_artifacts = release_artifacts[: request.max_release_artifacts]
    paths = _dedupe(base_artifacts + ledger.artifacts + verification.artifacts + release_artifacts)
    return [_artifact_hash(runs_dir, path) for path in paths], artifact_warning


def _artifact_hash(runs_dir: Path, rel_path: str) -> HandoffReleaseAttestationArtifactHash:
    root = runs_dir.resolve()
    path = (root / rel_path).resolve()
    kind = _artifact_kind(rel_path)
    if root != path and root not in path.parents:
        return HandoffReleaseAttestationArtifactHash(
            path=rel_path,
            kind=kind,
            status="unsafe",
            summary="Artifact path resolves outside runs_dir.",
        )
    if not path.exists() or path.is_dir():
        return HandoffReleaseAttestationArtifactHash(
            path=rel_path,
            kind=kind,
            status="missing",
            summary="Artifact is missing.",
        )
    return HandoffReleaseAttestationArtifactHash(
        path=rel_path,
        kind=kind,
        status="hashed",
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        summary="Artifact hash recorded.",
    )


def _artifact_kind(rel_path: str) -> str:
    if rel_path.startswith(f"{HANDOFF_LEDGER_DIR}/"):
        return "portfolio_control"
    if "/handoff_release_" in rel_path:
        return "release_control"
    if rel_path.endswith(".zip"):
        return "transfer_archive"
    return "release_artifact"


def _omitted_release_artifact_count(
    ledger: HandoffReleaseLedger,
    max_release_artifacts: int,
    *,
    include_release_artifacts: bool,
) -> int:
    if not include_release_artifacts:
        return 0
    release_artifacts: list[str] = []
    for item in ledger.items:
        release_artifacts.extend(item.artifacts)
    return max(0, len(_dedupe(release_artifacts)) - max_release_artifacts)


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


def _dedupe(items: list[str] | Any) -> list[str]:
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
