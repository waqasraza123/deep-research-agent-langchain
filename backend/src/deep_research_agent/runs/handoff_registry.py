"""Repository-level handoff readiness registry for completed run packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .contracts import ResearchRun
from .custody import read_run_custody_certificate
from .disclosure import read_run_disclosure_report
from .export_bundle import export_bundle_path, read_export_manifest
from .handoff import read_run_handoff_manifest
from .integrity import read_run_integrity_report
from .operator_audit import verify_operator_audit
from .retention import read_retention_policy

RegistryReadiness = Literal["ready_for_handoff", "needs_attention", "blocked"]

HANDOFF_REGISTRY_DIR = "_handoff"
HANDOFF_REGISTRY_JSON = "handoff_registry.json"
HANDOFF_REGISTRY_MD = "handoff_registry.md"

CONTROL_ARTIFACTS = (
    "retention_policy.json",
    "exports/export_manifest.json",
    "custody_certificate.json",
    "integrity_report.json",
    "disclosure_report.json",
    "handoff_manifest.json",
)


class HandoffRegistryRequest(BaseModel):
    requested_by: str = "operator"
    include_runs_without_handoff: bool = True
    require_operator_audit_valid: bool = True
    max_runs: int = Field(default=1000, ge=1, le=10000)
    notes: str = ""


class HandoffRegistryItem(BaseModel):
    thread_id: str
    run_status: str = "unknown"
    review_status: str = "unknown"
    created_at: str = ""
    updated_at: str = ""
    readiness: RegistryReadiness = "needs_attention"
    handoff_readiness: str = "missing"
    handoff_generated_at: str | None = None
    custody_readiness: str = "missing"
    integrity_readiness: str = "missing"
    disclosure_readiness: str = "missing"
    disclosure_risk_level: str = "unknown"
    retention_class: str = "missing"
    legal_hold: bool = False
    active_hold_ids: list[str] = Field(default_factory=list)
    export_present: bool = False
    export_profile: str = "missing"
    export_archive_sha256: str | None = None
    export_archive_valid: bool | None = None
    operator_audit_valid: bool = True
    operator_audit_event_count: int = 0
    missing_controls: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class HandoffRegistrySummary(BaseModel):
    total_runs: int = 0
    indexed_runs: int = 0
    excluded_runs: int = 0
    ready_for_handoff: int = 0
    needs_attention: int = 0
    blocked: int = 0
    missing_handoff: int = 0
    missing_controls: int = 0
    operator_audit_invalid: int = 0


class HandoffRegistry(BaseModel):
    registry_version: str = "1.0"
    generated_at: str
    requested_by: str = "operator"
    summary: HandoffRegistrySummary
    items: list[HandoffRegistryItem] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_registry(
    *,
    runs_dir: Path,
    runs: list[ResearchRun],
    request: HandoffRegistryRequest | None = None,
) -> HandoffRegistry:
    request = request or HandoffRegistryRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    limited_runs = runs[: request.max_runs]
    warnings: list[str] = []
    if len(runs) > request.max_runs:
        warnings.append(
            f"Registry indexed {request.max_runs} of {len(runs)} runs because max_runs was reached."
        )

    items: list[HandoffRegistryItem] = []
    excluded = len(runs) - len(limited_runs)
    for run in limited_runs:
        item = _registry_item(
            runs_dir,
            run,
            require_operator_audit_valid=request.require_operator_audit_valid,
        )
        if item.handoff_readiness == "missing" and not request.include_runs_without_handoff:
            excluded += 1
            continue
        items.append(item)

    summary = HandoffRegistrySummary(
        total_runs=len(runs),
        indexed_runs=len(items),
        excluded_runs=excluded,
        ready_for_handoff=sum(1 for item in items if item.readiness == "ready_for_handoff"),
        needs_attention=sum(1 for item in items if item.readiness == "needs_attention"),
        blocked=sum(1 for item in items if item.readiness == "blocked"),
        missing_handoff=sum(1 for item in items if item.handoff_readiness == "missing"),
        missing_controls=sum(1 for item in items if item.missing_controls),
        operator_audit_invalid=sum(1 for item in items if not item.operator_audit_valid),
    )
    registry = HandoffRegistry(
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        summary=summary,
        items=items,
        artifacts=[
            f"{HANDOFF_REGISTRY_DIR}/{HANDOFF_REGISTRY_JSON}",
            f"{HANDOFF_REGISTRY_DIR}/{HANDOFF_REGISTRY_MD}",
        ],
        warnings=warnings,
        notes=request.notes,
    )
    write_handoff_registry(runs_dir, registry)
    return registry


def read_handoff_registry(runs_dir: Path) -> HandoffRegistry:
    path = _registry_dir(runs_dir) / HANDOFF_REGISTRY_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffRegistry, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffRegistry.parse_obj(data)


def write_handoff_registry(runs_dir: Path, registry: HandoffRegistry) -> list[str]:
    registry_dir = _registry_dir(runs_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / HANDOFF_REGISTRY_JSON).write_text(
        json.dumps(_model_to_plain(registry), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (registry_dir / HANDOFF_REGISTRY_MD).write_text(
        render_handoff_registry_markdown(registry),
        encoding="utf-8",
    )
    return [
        f"{HANDOFF_REGISTRY_DIR}/{HANDOFF_REGISTRY_JSON}",
        f"{HANDOFF_REGISTRY_DIR}/{HANDOFF_REGISTRY_MD}",
    ]


def render_handoff_registry_markdown(registry: HandoffRegistry) -> str:
    lines = [
        "# Handoff Registry",
        "",
        f"- Generated at: `{registry.generated_at}`",
        f"- Requested by: `{registry.requested_by}`",
        f"- Total runs: {registry.summary.total_runs}",
        f"- Indexed runs: {registry.summary.indexed_runs}",
        f"- Ready for handoff: {registry.summary.ready_for_handoff}",
        f"- Needs attention: {registry.summary.needs_attention}",
        f"- Blocked: {registry.summary.blocked}",
        f"- Missing handoff manifests: {registry.summary.missing_handoff}",
        f"- Runs with missing controls: {registry.summary.missing_controls}",
        f"- Runs with invalid operator audit: {registry.summary.operator_audit_invalid}",
        "",
        "## Runs",
        "",
    ]
    if not registry.items:
        lines.extend(["No runs were indexed.", ""])
    for item in registry.items:
        lines.extend(
            [
                f"### {item.thread_id}",
                "",
                f"- Readiness: `{item.readiness}`",
                f"- Run status: `{item.run_status}`",
                f"- Review status: `{item.review_status}`",
                f"- Handoff: `{item.handoff_readiness}`",
                f"- Custody: `{item.custody_readiness}`",
                f"- Integrity: `{item.integrity_readiness}`",
                f"- Disclosure: `{item.disclosure_readiness}` risk `{item.disclosure_risk_level}`",
                f"- Retention: `{item.retention_class}`; legal hold: `{item.legal_hold}`",
                f"- Export: `{item.export_profile}`; hash valid: `{item.export_archive_valid}`",
                f"- Operator audit valid: `{item.operator_audit_valid}`",
                "- Missing controls: "
                + (", ".join(f"`{control}`" for control in item.missing_controls) or "none"),
                "- Blockers: " + (", ".join(item.blockers) or "none"),
                "- Warnings: " + (", ".join(item.warnings) or "none"),
                "",
            ]
        )
    if registry.warnings:
        lines.extend(["## Registry Warnings", ""])
        lines.extend(f"- {warning}" for warning in registry.warnings)
        lines.append("")
    if registry.notes:
        lines.extend(["## Notes", "", registry.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _registry_item(
    runs_dir: Path,
    run: ResearchRun,
    *,
    require_operator_audit_valid: bool,
) -> HandoffRegistryItem:
    thread_id = run.thread_id
    missing_controls: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    artifacts: list[str] = []
    run_dir = runs_dir / thread_id

    handoff_readiness = "missing"
    handoff_generated_at: str | None = None
    try:
        handoff = read_run_handoff_manifest(runs_dir, thread_id)
        handoff_readiness = handoff.readiness
        handoff_generated_at = handoff.generated_at
        artifacts.extend(["handoff_manifest.json", "handoff_manifest.md"])
        blockers.extend(handoff.blockers)
        warnings.extend(handoff.warnings)
    except FileNotFoundError:
        missing_controls.append("handoff_manifest")
        blockers.append("Handoff manifest is missing.")
    except ValueError as e:
        blockers.append(f"Handoff manifest could not be read: {e}")

    retention_class = "missing"
    legal_hold = False
    active_hold_ids: list[str] = []
    try:
        retention = read_retention_policy(runs_dir, thread_id)
        retention_class = retention.retention_class
        legal_hold = retention.legal_hold
        active_hold_ids = [hold.hold_id for hold in retention.active_holds]
        artifacts.extend(["retention_policy.json", "retention_policy.md"])
    except FileNotFoundError:
        missing_controls.append("retention_policy")
    except ValueError as e:
        blockers.append(f"Retention policy could not be read: {e}")

    export_present = False
    export_profile = "missing"
    export_archive_sha256: str | None = None
    export_archive_valid: bool | None = None
    try:
        export_manifest = read_export_manifest(runs_dir, thread_id)
        archive_path = export_bundle_path(runs_dir, thread_id)
        actual_sha = file_sha256(archive_path)
        export_present = True
        export_profile = export_manifest.profile
        export_archive_sha256 = export_manifest.archive_sha256 or actual_sha
        export_archive_valid = (
            not export_manifest.archive_sha256 or actual_sha == export_manifest.archive_sha256
        )
        artifacts.extend(
            [
                "exports/run_export.zip",
                "exports/export_manifest.json",
                "exports/export_manifest.md",
            ]
        )
        if export_archive_valid is False:
            blockers.append("Export archive hash does not match export manifest.")
    except FileNotFoundError:
        missing_controls.append("export_bundle")
        export_archive_valid = False
    except ValueError as e:
        export_archive_valid = False
        blockers.append(f"Export manifest could not be read: {e}")

    custody_readiness = "missing"
    try:
        custody = read_run_custody_certificate(runs_dir, thread_id)
        custody_readiness = custody.readiness
        artifacts.extend(["custody_certificate.json", "custody_certificate.md"])
        if custody.readiness == "blocked":
            blockers.extend(custody.blockers)
        elif custody.readiness != "ready":
            warnings.extend(custody.warnings)
    except FileNotFoundError:
        missing_controls.append("custody_certificate")
    except ValueError as e:
        blockers.append(f"Custody certificate could not be read: {e}")

    integrity_readiness = "missing"
    try:
        integrity = read_run_integrity_report(runs_dir, thread_id)
        integrity_readiness = integrity.readiness
        artifacts.extend(["integrity_report.json", "integrity_report.md"])
        if integrity.readiness == "failed":
            blockers.extend(integrity.failures)
        elif integrity.readiness != "valid":
            warnings.extend(integrity.warnings)
    except FileNotFoundError:
        missing_controls.append("integrity_report")
    except ValueError as e:
        blockers.append(f"Integrity report could not be read: {e}")

    disclosure_readiness = "missing"
    disclosure_risk_level = "unknown"
    try:
        disclosure = read_run_disclosure_report(runs_dir, thread_id)
        disclosure_readiness = disclosure.readiness
        disclosure_risk_level = disclosure.risk_level
        artifacts.extend(["disclosure_report.json", "disclosure_report.md"])
        if disclosure.readiness == "blocked":
            blockers.append(
                "Disclosure report has "
                f"{disclosure.high_or_critical_count} high or critical findings."
            )
        elif disclosure.readiness != "clear":
            warnings.append(
                f"Disclosure report requires review with risk `{disclosure.risk_level}`."
            )
    except FileNotFoundError:
        missing_controls.append("disclosure_report")
    except ValueError as e:
        blockers.append(f"Disclosure report could not be read: {e}")

    audit_valid = True
    audit_event_count = 0
    try:
        audit = verify_operator_audit(runs_dir, thread_id)
        audit_valid = audit.valid
        audit_event_count = audit.event_count
        artifacts.extend(["operator_audit.jsonl", "operator_audit.md"])
    except ValueError as e:
        audit_valid = False
        blockers.append(f"Operator audit could not be verified: {e}")
    if require_operator_audit_valid and not audit_valid:
        blockers.append("Operator audit hash-chain verification failed.")
    elif not audit_valid:
        warnings.append("Operator audit hash-chain verification failed.")

    if _handoff_stale(run_dir, handoff_generated_at):
        warnings.append("Handoff manifest is older than one or more upstream control artifacts.")

    status = _enum_value(run.status)
    review_status = _enum_value(run.review.status)
    if status in {"failed", "cancelled"}:
        blockers.append(f"Run is `{status}`.")
    if review_status in {"changes_requested", "rejected"}:
        blockers.append(f"Review status is `{review_status}`.")

    readiness: RegistryReadiness = "ready_for_handoff"
    if blockers or (missing_controls and handoff_readiness == "missing"):
        readiness = "blocked"
    elif handoff_readiness == "blocked":
        readiness = "blocked"
    elif warnings or missing_controls or handoff_readiness != "ready_for_handoff":
        readiness = "needs_attention"

    return HandoffRegistryItem(
        thread_id=thread_id,
        run_status=status,
        review_status=review_status,
        created_at=_datetime_to_iso(run.created_at),
        updated_at=_datetime_to_iso(run.updated_at),
        readiness=readiness,
        handoff_readiness=handoff_readiness,
        handoff_generated_at=handoff_generated_at,
        custody_readiness=custody_readiness,
        integrity_readiness=integrity_readiness,
        disclosure_readiness=disclosure_readiness,
        disclosure_risk_level=disclosure_risk_level,
        retention_class=retention_class,
        legal_hold=legal_hold,
        active_hold_ids=active_hold_ids,
        export_present=export_present,
        export_profile=export_profile,
        export_archive_sha256=export_archive_sha256,
        export_archive_valid=export_archive_valid,
        operator_audit_valid=audit_valid,
        operator_audit_event_count=audit_event_count,
        missing_controls=sorted(set(missing_controls)),
        blockers=_dedupe(blockers),
        warnings=_dedupe(warnings),
        artifacts=sorted(set(artifacts)),
    )


def _handoff_stale(run_dir: Path, handoff_generated_at: str | None) -> bool:
    handoff_path = run_dir / "handoff_manifest.json"
    if handoff_generated_at is None or not handoff_path.exists():
        return False
    try:
        handoff_mtime = handoff_path.stat().st_mtime
    except OSError:
        return False
    for rel_path in CONTROL_ARTIFACTS:
        if rel_path == "handoff_manifest.json":
            continue
        path = run_dir / rel_path
        try:
            if path.exists() and path.stat().st_mtime > handoff_mtime:
                return True
        except OSError:
            continue
    return False


def _registry_dir(runs_dir: Path) -> Path:
    root = runs_dir.resolve()
    registry_dir = (root / HANDOFF_REGISTRY_DIR).resolve()
    if root != registry_dir and root not in registry_dir.parents:
        raise ValueError("Invalid registry directory")
    return registry_dir


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _datetime_to_iso(value: Any) -> str:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value or "")


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
