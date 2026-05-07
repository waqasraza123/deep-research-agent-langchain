"""Verification reports for handoff release portfolio attestations."""

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
    HANDOFF_RELEASE_ATTESTATION_MD,
    HandoffReleaseAttestation,
    HandoffReleaseAttestationArtifactHash,
    read_handoff_release_attestation,
    render_handoff_release_attestation_markdown,
)
from .handoff_release_ledger import (
    HANDOFF_LEDGER_DIR,
    HANDOFF_RELEASE_LEDGER_JSON,
    read_handoff_release_ledger,
)
from .handoff_release_ledger_verification import (
    HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON,
    read_handoff_release_ledger_verification_report,
)
from .operator_audit import verify_operator_audit

AttestationVerificationStatus = Literal["passed", "warning", "failed"]
AttestationVerificationReadiness = Literal["valid", "warnings", "failed"]

HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON = (
    "handoff_release_attestation_verification.json"
)
HANDOFF_RELEASE_ATTESTATION_VERIFICATION_MD = "handoff_release_attestation_verification.md"


class HandoffReleaseAttestationVerificationRequest(BaseModel):
    requested_by: str = "operator"
    require_attestation_artifacts: bool = True
    require_attestation_ready: bool = True
    require_control_hashes: bool = True
    require_artifact_hashes: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleaseAttestationVerificationFinding(BaseModel):
    finding_id: str
    title: str
    status: AttestationVerificationStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleaseAttestationArtifactVerification(BaseModel):
    path: str
    status: AttestationVerificationStatus
    expected_status: str = ""
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    expected_size_bytes: int | None = None
    actual_size_bytes: int | None = None
    summary: str = ""


class HandoffReleaseAttestationVerificationReport(BaseModel):
    report_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    readiness: AttestationVerificationReadiness = "warnings"
    attestation_generated_at: str
    attestation_sha256: str | None = None
    current_ledger_sha256: str | None = None
    current_ledger_verification_sha256: str | None = None
    findings: list[HandoffReleaseAttestationVerificationFinding] = Field(default_factory=list)
    artifact_verifications: list[HandoffReleaseAttestationArtifactVerification] = Field(
        default_factory=list
    )
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_attestation_verification_report(
    *,
    runs_dir: Path,
    request: HandoffReleaseAttestationVerificationRequest | None = None,
    attestation: HandoffReleaseAttestation | None = None,
) -> HandoffReleaseAttestationVerificationReport:
    request = request or HandoffReleaseAttestationVerificationRequest()
    attestation = attestation or read_handoff_release_attestation(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    ledger_dir = _ledger_dir(runs_dir)
    artifact_verifications = _artifact_verifications(runs_dir, attestation)
    findings = [
        _attestation_artifacts_finding(
            runs_dir,
            attestation,
            required=request.require_attestation_artifacts,
        ),
        _attestation_readiness_finding(
            attestation,
            required=request.require_attestation_ready,
        ),
        _control_hashes_finding(
            runs_dir,
            attestation,
            required=request.require_control_hashes,
        ),
        _artifact_hashes_finding(
            artifact_verifications,
            required=request.require_artifact_hashes,
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
    readiness: AttestationVerificationReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"

    report = HandoffReleaseAttestationVerificationReport(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        attestation_generated_at=attestation.generated_at,
        attestation_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_ATTESTATION_JSON
        ),
        current_ledger_sha256=_file_sha256_or_none(ledger_dir / HANDOFF_RELEASE_LEDGER_JSON),
        current_ledger_verification_sha256=_file_sha256_or_none(
            ledger_dir / HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON
        ),
        findings=findings,
        artifact_verifications=artifact_verifications,
        failures=failures,
        warnings=warnings,
        required_controls={
            "attestation_artifacts": request.require_attestation_artifacts,
            "attestation_ready": request.require_attestation_ready,
            "control_hashes": request.require_control_hashes,
            "artifact_hashes": request.require_artifact_hashes,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_attestation_verification_report(runs_dir, report)
    return report


def read_handoff_release_attestation_verification_report(
    runs_dir: Path,
) -> HandoffReleaseAttestationVerificationReport:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseAttestationVerificationReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseAttestationVerificationReport.parse_obj(data)


def write_handoff_release_attestation_verification_report(
    runs_dir: Path,
    report: HandoffReleaseAttestationVerificationReport,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_ATTESTATION_VERIFICATION_MD).write_text(
        render_handoff_release_attestation_verification_markdown(report),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_ATTESTATION_VERIFICATION_MD}",
    ]


def render_handoff_release_attestation_verification_markdown(
    report: HandoffReleaseAttestationVerificationReport,
) -> str:
    lines = [
        "# Handoff Release Attestation Verification",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Attestation generated at: `{report.attestation_generated_at}`",
        f"- Attestation SHA-256: `{report.attestation_sha256 or 'unavailable'}`",
        f"- Current ledger SHA-256: `{report.current_ledger_sha256 or 'unavailable'}`",
        "- Current ledger verification SHA-256: "
        f"`{report.current_ledger_verification_sha256 or 'unavailable'}`",
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
                    "- Expected size bytes: "
                    f"{_display_size(item.expected_size_bytes)}",
                    "- Actual size bytes: "
                    f"{_display_size(item.actual_size_bytes)}",
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


def _attestation_artifacts_finding(
    runs_dir: Path,
    attestation: HandoffReleaseAttestation,
    *,
    required: bool,
) -> HandoffReleaseAttestationVerificationFinding:
    ledger_dir = _ledger_dir(runs_dir)
    expected = [HANDOFF_RELEASE_ATTESTATION_JSON, HANDOFF_RELEASE_ATTESTATION_MD]
    missing = [name for name in expected if not (ledger_dir / name).exists()]
    if missing:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="attestation_artifacts",
            title="Attestation Artifacts",
            status="failed" if required else "warning",
            summary="Attestation artifacts are missing: " + ", ".join(missing),
            required_action="Regenerate or restore the handoff release attestation.",
            evidence_artifacts=attestation.artifacts,
            metadata={"missing": missing, "required": required},
        )
    rendered = render_handoff_release_attestation_markdown(attestation)
    current_md = (ledger_dir / HANDOFF_RELEASE_ATTESTATION_MD).read_text(encoding="utf-8")
    if rendered != current_md:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="attestation_artifacts",
            title="Attestation Artifacts",
            status="failed" if required else "warning",
            summary="Attestation Markdown sidecar does not match the JSON attestation content.",
            required_action="Regenerate the handoff release attestation sidecars.",
            evidence_artifacts=attestation.artifacts,
            metadata={"required": required},
        )
    return HandoffReleaseAttestationVerificationFinding(
        finding_id="attestation_artifacts",
        title="Attestation Artifacts",
        status="passed",
        summary="Attestation JSON and Markdown artifacts are mutually consistent.",
        evidence_artifacts=attestation.artifacts,
        metadata={"required": required},
    )


def _attestation_readiness_finding(
    attestation: HandoffReleaseAttestation,
    *,
    required: bool,
) -> HandoffReleaseAttestationVerificationFinding:
    if attestation.readiness != "attested":
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="attestation_readiness",
            title="Attestation Readiness",
            status="failed" if required else "warning",
            summary=f"Attestation readiness is {attestation.readiness}.",
            required_action="Resolve attestation blockers before relying on verification.",
            evidence_artifacts=attestation.artifacts,
            metadata={
                "readiness": attestation.readiness,
                "blockers": attestation.blockers,
                "warnings": attestation.warnings,
                "required": required,
            },
        )
    return HandoffReleaseAttestationVerificationFinding(
        finding_id="attestation_readiness",
        title="Attestation Readiness",
        status="passed",
        summary="Saved attestation readiness is attested.",
        evidence_artifacts=attestation.artifacts,
        metadata={"readiness": attestation.readiness, "required": required},
    )


def _control_hashes_finding(
    runs_dir: Path,
    attestation: HandoffReleaseAttestation,
    *,
    required: bool,
) -> HandoffReleaseAttestationVerificationFinding:
    ledger_dir = _ledger_dir(runs_dir)
    current_ledger_sha = _file_sha256_or_none(ledger_dir / HANDOFF_RELEASE_LEDGER_JSON)
    current_verification_sha = _file_sha256_or_none(
        ledger_dir / HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON
    )
    drift: list[str] = []
    if current_ledger_sha != attestation.ledger_sha256:
        drift.append("ledger_sha256")
    if current_verification_sha != attestation.ledger_verification_sha256:
        drift.append("ledger_verification_sha256")
    try:
        ledger = read_handoff_release_ledger(runs_dir)
        if ledger.generated_at != attestation.ledger_generated_at:
            drift.append("ledger_generated_at")
    except FileNotFoundError:
        drift.append("ledger_missing")
    except Exception as e:
        drift.append(f"ledger_unreadable:{type(e).__name__}")
    try:
        verification = read_handoff_release_ledger_verification_report(runs_dir)
        if verification.generated_at != attestation.ledger_verification_generated_at:
            drift.append("ledger_verification_generated_at")
    except FileNotFoundError:
        drift.append("ledger_verification_missing")
    except Exception as e:
        drift.append(f"ledger_verification_unreadable:{type(e).__name__}")
    if drift:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="control_hashes",
            title="Control Hashes",
            status="failed" if required else "warning",
            summary="Current ledger controls differ from the saved attestation.",
            required_action="Regenerate ledger controls or regenerate the attestation.",
            evidence_artifacts=[
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_JSON}",
                f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON}",
            ],
            metadata={
                "drift": drift,
                "expected_ledger_sha256": attestation.ledger_sha256,
                "actual_ledger_sha256": current_ledger_sha,
                "expected_ledger_verification_sha256": (
                    attestation.ledger_verification_sha256
                ),
                "actual_ledger_verification_sha256": current_verification_sha,
                "required": required,
            },
        )
    return HandoffReleaseAttestationVerificationFinding(
        finding_id="control_hashes",
        title="Control Hashes",
        status="passed",
        summary="Current ledger controls match the saved attestation hashes.",
        evidence_artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON}",
        ],
        metadata={"required": required},
    )


def _artifact_hashes_finding(
    verifications: list[HandoffReleaseAttestationArtifactVerification],
    *,
    required: bool,
) -> HandoffReleaseAttestationVerificationFinding:
    failed = [item.path for item in verifications if item.status == "failed"]
    warnings = [item.path for item in verifications if item.status == "warning"]
    if failed:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="artifact_hashes",
            title="Artifact Hashes",
            status="failed" if required else "warning",
            summary="One or more attested artifacts failed hash verification.",
            required_action="Restore changed artifacts or regenerate the attestation.",
            evidence_artifacts=[item.path for item in verifications],
            metadata={"failed": failed, "warnings": warnings, "required": required},
        )
    if warnings:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="artifact_hashes",
            title="Artifact Hashes",
            status="warning",
            summary="One or more attested artifacts have verification warnings.",
            required_action="Review artifact warnings before relying on verification.",
            evidence_artifacts=[item.path for item in verifications],
            metadata={"warnings": warnings, "required": required},
        )
    return HandoffReleaseAttestationVerificationFinding(
        finding_id="artifact_hashes",
        title="Artifact Hashes",
        status="passed",
        summary="Every attested artifact hash matches current disk state.",
        evidence_artifacts=[item.path for item in verifications],
        metadata={"artifact_count": len(verifications), "required": required},
    )


def _operator_audit_finding(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleaseAttestationVerificationFinding:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleaseAttestationVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before verification reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleaseAttestationVerificationFinding(
        finding_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _artifact_verifications(
    runs_dir: Path,
    attestation: HandoffReleaseAttestation,
) -> list[HandoffReleaseAttestationArtifactVerification]:
    return [
        _artifact_verification(runs_dir, artifact)
        for artifact in attestation.artifact_hashes
    ]


def _artifact_verification(
    runs_dir: Path,
    artifact: HandoffReleaseAttestationArtifactHash,
) -> HandoffReleaseAttestationArtifactVerification:
    root = runs_dir.resolve()
    path = (root / artifact.path).resolve()
    if root != path and root not in path.parents:
        return HandoffReleaseAttestationArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            expected_size_bytes=artifact.size_bytes,
            summary="Artifact path resolves outside runs_dir.",
        )
    if not path.exists() or path.is_dir():
        return HandoffReleaseAttestationArtifactVerification(
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
        return HandoffReleaseAttestationArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            actual_sha256=actual_sha256,
            expected_size_bytes=artifact.size_bytes,
            actual_size_bytes=actual_size,
            summary="Artifact exists now but was not hashed in the saved attestation.",
        )
    if actual_sha256 != artifact.sha256 or actual_size != artifact.size_bytes:
        return HandoffReleaseAttestationArtifactVerification(
            path=artifact.path,
            status="failed",
            expected_status=artifact.status,
            expected_sha256=artifact.sha256,
            actual_sha256=actual_sha256,
            expected_size_bytes=artifact.size_bytes,
            actual_size_bytes=actual_size,
            summary="Artifact hash or size differs from the saved attestation.",
        )
    return HandoffReleaseAttestationArtifactVerification(
        path=artifact.path,
        status="passed",
        expected_status=artifact.status,
        expected_sha256=artifact.sha256,
        actual_sha256=actual_sha256,
        expected_size_bytes=artifact.size_bytes,
        actual_size_bytes=actual_size,
        summary="Artifact matches the saved attestation.",
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
