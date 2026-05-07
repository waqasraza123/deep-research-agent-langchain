"""Verification reports for repository-level handoff release custody ledgers."""

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
    HandoffReleaseLedgerItem,
    HandoffReleaseLedgerRequest,
    read_handoff_release_ledger,
    render_handoff_release_ledger_markdown,
    snapshot_handoff_release_ledger,
)
from .operator_audit import verify_operator_audit

LedgerVerificationStatus = Literal["passed", "warning", "failed"]
LedgerVerificationReadiness = Literal["valid", "warnings", "failed"]

HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON = "handoff_release_ledger_verification.json"
HANDOFF_RELEASE_LEDGER_VERIFICATION_MD = "handoff_release_ledger_verification.md"


class HandoffReleaseLedgerVerificationRequest(BaseModel):
    requested_by: str = "operator"
    require_ledger_artifacts: bool = True
    require_snapshot_match: bool = True
    require_artifact_presence: bool = True
    require_global_operator_audit: bool = True
    notes: str = ""


class HandoffReleaseLedgerVerificationFinding(BaseModel):
    finding_id: str
    title: str
    status: LedgerVerificationStatus
    summary: str = ""
    required_action: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffReleaseLedgerItemVerification(BaseModel):
    release_id: str
    status: LedgerVerificationStatus = "warning"
    summary: str = ""
    changed_fields: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)


class HandoffReleaseLedgerVerificationReport(BaseModel):
    report_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    readiness: LedgerVerificationReadiness = "warnings"
    ledger_generated_at: str
    ledger_sha256: str | None = None
    current_snapshot_generated_at: str
    findings: list[HandoffReleaseLedgerVerificationFinding] = Field(default_factory=list)
    item_verifications: list[HandoffReleaseLedgerItemVerification] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_ledger_verification_report(
    *,
    runs_dir: Path,
    request: HandoffReleaseLedgerVerificationRequest | None = None,
    ledger: HandoffReleaseLedger | None = None,
) -> HandoffReleaseLedgerVerificationReport:
    request = request or HandoffReleaseLedgerVerificationRequest()
    ledger = ledger or read_handoff_release_ledger(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    current = snapshot_handoff_release_ledger(
        runs_dir=runs_dir,
        request=_request_from_ledger(ledger),
    )
    ledger_path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_LEDGER_JSON

    findings = [
        _ledger_artifacts_finding(
            runs_dir,
            ledger,
            required=request.require_ledger_artifacts,
        ),
        _summary_finding(
            ledger,
            current,
            required=request.require_snapshot_match,
        ),
        _operator_audit_finding(
            runs_dir,
            required=request.require_global_operator_audit,
        ),
    ]
    item_verifications = _item_verifications(
        runs_dir,
        ledger,
        current,
        require_snapshot_match=request.require_snapshot_match,
        require_artifact_presence=request.require_artifact_presence,
    )
    findings.append(
        _item_inventory_finding(
            item_verifications,
            required=request.require_snapshot_match or request.require_artifact_presence,
        )
    )

    failures = [finding.summary for finding in findings if finding.status == "failed"]
    failures.extend(item.summary for item in item_verifications if item.status == "failed")
    warnings = [finding.summary for finding in findings if finding.status == "warning"]
    warnings.extend(item.summary for item in item_verifications if item.status == "warning")
    failures = _dedupe(failures)
    warnings = _dedupe(warnings)
    readiness: LedgerVerificationReadiness = "valid"
    if failures:
        readiness = "failed"
    elif warnings:
        readiness = "warnings"

    report = HandoffReleaseLedgerVerificationReport(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        ledger_generated_at=ledger.generated_at,
        ledger_sha256=_file_sha256_or_none(ledger_path),
        current_snapshot_generated_at=current.generated_at,
        findings=findings,
        item_verifications=item_verifications,
        failures=failures,
        warnings=warnings,
        required_controls={
            "ledger_artifacts": request.require_ledger_artifacts,
            "snapshot_match": request.require_snapshot_match,
            "artifact_presence": request.require_artifact_presence,
            "global_operator_audit": request.require_global_operator_audit,
        },
        artifacts=[
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON}",
            f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_MD}",
        ],
        notes=request.notes,
    )
    write_handoff_release_ledger_verification_report(runs_dir, report)
    return report


def read_handoff_release_ledger_verification_report(
    runs_dir: Path,
) -> HandoffReleaseLedgerVerificationReport:
    path = _ledger_dir(runs_dir) / HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseLedgerVerificationReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseLedgerVerificationReport.parse_obj(data)


def write_handoff_release_ledger_verification_report(
    runs_dir: Path,
    report: HandoffReleaseLedgerVerificationReport,
) -> list[str]:
    ledger_dir = _ledger_dir(runs_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ledger_dir / HANDOFF_RELEASE_LEDGER_VERIFICATION_MD).write_text(
        render_handoff_release_ledger_verification_markdown(report),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_JSON}",
        f"{HANDOFF_LEDGER_DIR}/{HANDOFF_RELEASE_LEDGER_VERIFICATION_MD}",
    ]


def render_handoff_release_ledger_verification_markdown(
    report: HandoffReleaseLedgerVerificationReport,
) -> str:
    lines = [
        "# Handoff Release Ledger Verification",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Ledger generated at: `{report.ledger_generated_at}`",
        f"- Ledger SHA-256: `{report.ledger_sha256 or 'unavailable'}`",
        f"- Current snapshot generated at: `{report.current_snapshot_generated_at}`",
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
    if report.item_verifications:
        lines.extend(["## Release Items", ""])
        for item in report.item_verifications:
            lines.extend(
                [
                    f"### {item.release_id}",
                    "",
                    f"- Status: `{item.status}`",
                    f"- Summary: {item.summary or 'None'}",
                    "- Changed fields: "
                    + (", ".join(f"`{field}`" for field in item.changed_fields) or "none"),
                    "- Missing artifacts: "
                    + (", ".join(f"`{path}`" for path in item.missing_artifacts) or "none"),
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


def _ledger_artifacts_finding(
    runs_dir: Path,
    ledger: HandoffReleaseLedger,
    *,
    required: bool,
) -> HandoffReleaseLedgerVerificationFinding:
    expected = [HANDOFF_RELEASE_LEDGER_JSON, HANDOFF_RELEASE_LEDGER_MD]
    ledger_dir = _ledger_dir(runs_dir)
    missing = [name for name in expected if not (ledger_dir / name).exists()]
    if missing:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="ledger_artifacts",
            title="Ledger Artifacts",
            status="failed" if required else "warning",
            summary="Ledger artifacts are missing: " + ", ".join(missing),
            required_action="Regenerate or restore the handoff release ledger.",
            evidence_artifacts=ledger.artifacts,
            metadata={"missing": missing, "required": required},
        )
    rendered = render_handoff_release_ledger_markdown(ledger)
    current_md = (ledger_dir / HANDOFF_RELEASE_LEDGER_MD).read_text(encoding="utf-8")
    if rendered != current_md:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="ledger_artifacts",
            title="Ledger Artifacts",
            status="failed" if required else "warning",
            summary="Ledger Markdown sidecar does not match the JSON ledger content.",
            required_action="Regenerate the handoff release ledger sidecars.",
            evidence_artifacts=ledger.artifacts,
            metadata={"required": required},
        )
    return HandoffReleaseLedgerVerificationFinding(
        finding_id="ledger_artifacts",
        title="Ledger Artifacts",
        status="passed",
        summary="Ledger JSON and Markdown artifacts exist and are mutually consistent.",
        evidence_artifacts=ledger.artifacts,
        metadata={"required": required},
    )


def _summary_finding(
    ledger: HandoffReleaseLedger,
    current: HandoffReleaseLedger,
    *,
    required: bool,
) -> HandoffReleaseLedgerVerificationFinding:
    expected = _summary_payload(ledger)
    actual = _summary_payload(current)
    if expected != actual:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="ledger_summary",
            title="Ledger Summary",
            status="failed" if required else "warning",
            summary="Current release custody summary differs from the saved ledger.",
            required_action="Regenerate the ledger after reviewing custody changes.",
            evidence_artifacts=ledger.artifacts,
            metadata={
                "saved_summary": expected,
                "current_summary": actual,
                "required": required,
            },
        )
    return HandoffReleaseLedgerVerificationFinding(
        finding_id="ledger_summary",
        title="Ledger Summary",
        status="passed",
        summary="Current release custody summary matches the saved ledger.",
        evidence_artifacts=ledger.artifacts,
        metadata={"required": required},
    )


def _operator_audit_finding(
    runs_dir: Path,
    *,
    required: bool,
) -> HandoffReleaseLedgerVerificationFinding:
    try:
        verification = verify_operator_audit(runs_dir)
    except ValueError as e:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary=f"Global operator audit could not be verified: {e}",
            required_action="Repair or investigate the global operator audit log.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata={"required": required},
        )
    if not verification.valid:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="global_operator_audit",
            title="Global Operator Audit",
            status="failed" if required else "warning",
            summary="Global operator audit hash-chain verification failed.",
            required_action="Investigate audit log corruption before ledger reliance.",
            evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
            metadata=_model_to_plain(verification) | {"required": required},
        )
    return HandoffReleaseLedgerVerificationFinding(
        finding_id="global_operator_audit",
        title="Global Operator Audit",
        status="passed",
        summary="Global operator audit hash chain verifies.",
        evidence_artifacts=["_audit/operator_audit.jsonl", "_audit/operator_audit.md"],
        metadata=_model_to_plain(verification) | {"required": required},
    )


def _item_verifications(
    runs_dir: Path,
    ledger: HandoffReleaseLedger,
    current: HandoffReleaseLedger,
    *,
    require_snapshot_match: bool,
    require_artifact_presence: bool,
) -> list[HandoffReleaseLedgerItemVerification]:
    current_by_id = {item.release_id: item for item in current.items}
    saved_ids = {item.release_id for item in ledger.items}
    current_ids = set(current_by_id)
    out: list[HandoffReleaseLedgerItemVerification] = []
    for item in ledger.items:
        missing_artifacts = _missing_artifacts(runs_dir, item.artifacts)
        changed_fields: list[str] = []
        current_item = current_by_id.get(item.release_id)
        if current_item is None:
            changed_fields.append("release_missing_from_current_snapshot")
        else:
            changed_fields = _changed_item_fields(item, current_item)
        status: LedgerVerificationStatus = "passed"
        summary = "Ledger item matches current release custody state."
        if changed_fields and require_snapshot_match:
            status = "failed"
            summary = f"{item.release_id}: saved ledger item differs from current custody state."
        elif changed_fields:
            status = "warning"
            summary = f"{item.release_id}: saved ledger item differs from current custody state."
        if missing_artifacts and require_artifact_presence:
            status = "failed"
            summary = f"{item.release_id}: ledger references missing artifacts."
        elif missing_artifacts and status == "passed":
            status = "warning"
            summary = f"{item.release_id}: ledger references missing artifacts."
        out.append(
            HandoffReleaseLedgerItemVerification(
                release_id=item.release_id,
                status=status,
                summary=summary,
                changed_fields=changed_fields,
                missing_artifacts=missing_artifacts,
            )
        )
    for release_id in sorted(current_ids - saved_ids):
        out.append(
            HandoffReleaseLedgerItemVerification(
                release_id=release_id,
                status="failed" if require_snapshot_match else "warning",
                summary=f"{release_id}: current release is missing from the saved ledger.",
                changed_fields=["release_missing_from_saved_ledger"],
            )
        )
    return out


def _item_inventory_finding(
    items: list[HandoffReleaseLedgerItemVerification],
    *,
    required: bool,
) -> HandoffReleaseLedgerVerificationFinding:
    failed = [item.release_id for item in items if item.status == "failed"]
    warnings = [item.release_id for item in items if item.status == "warning"]
    if failed:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="ledger_items",
            title="Ledger Item Inventory",
            status="failed" if required else "warning",
            summary="One or more ledger items failed verification.",
            required_action="Regenerate the ledger after resolving release custody drift.",
            metadata={"failed_release_ids": failed, "warning_release_ids": warnings},
        )
    if warnings:
        return HandoffReleaseLedgerVerificationFinding(
            finding_id="ledger_items",
            title="Ledger Item Inventory",
            status="warning",
            summary="One or more ledger items have verification warnings.",
            required_action="Review ledger item warnings before portfolio custody reliance.",
            metadata={"warning_release_ids": warnings},
        )
    return HandoffReleaseLedgerVerificationFinding(
        finding_id="ledger_items",
        title="Ledger Item Inventory",
        status="passed",
        summary="Every ledger item matches current release custody state.",
        metadata={"verified_items": len(items), "required": required},
    )


def _changed_item_fields(
    saved: HandoffReleaseLedgerItem,
    current: HandoffReleaseLedgerItem,
) -> list[str]:
    fields = [
        "readiness",
        "release_readiness",
        "release_verification_readiness",
        "bundle_readiness",
        "bundle_archive_sha256",
        "bundle_archive_valid",
        "bundle_verification_readiness",
        "receipt_readiness",
        "receipt_outcome",
        "recipient_bundle_sha256",
        "recipient_checksum_valid",
        "operator_audit_valid",
        "missing_controls",
        "blockers",
        "warnings",
    ]
    changed = []
    for field in fields:
        if getattr(saved, field) != getattr(current, field):
            changed.append(field)
    return changed


def _missing_artifacts(runs_dir: Path, artifacts: list[str]) -> list[str]:
    root = runs_dir.resolve()
    missing = []
    for rel_path in artifacts:
        path = (root / rel_path).resolve()
        if root != path and root not in path.parents:
            missing.append(rel_path)
            continue
        if not path.exists():
            missing.append(rel_path)
    return sorted(set(missing))


def _request_from_ledger(ledger: HandoffReleaseLedger) -> HandoffReleaseLedgerRequest:
    controls = ledger.required_controls
    max_releases = controls.get("max_releases", ledger.summary.total_releases)
    try:
        max_releases_int = int(max_releases)
    except (TypeError, ValueError):
        max_releases_int = max(ledger.summary.total_releases, len(ledger.items), 1)
    return HandoffReleaseLedgerRequest(
        requested_by=ledger.requested_by,
        include_releases_without_receipt=bool(
            controls.get("include_releases_without_receipt", True)
        ),
        require_release_ready=bool(controls.get("release_ready", True)),
        require_release_verification_valid=bool(
            controls.get("release_verification_valid", True)
        ),
        require_bundle_ready=bool(controls.get("bundle_ready", True)),
        require_bundle_verification_valid=bool(
            controls.get("bundle_verification_valid", True)
        ),
        require_receipt_recorded=bool(controls.get("receipt_recorded", True)),
        require_recipient_checksum_match=bool(
            controls.get("recipient_checksum_match", True)
        ),
        require_global_operator_audit=bool(controls.get("global_operator_audit", True)),
        max_releases=min(max(max_releases_int, 1), 10000),
        notes=ledger.notes,
    )


def _summary_payload(ledger: HandoffReleaseLedger) -> dict[str, int]:
    return _model_to_plain(ledger.summary)


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
