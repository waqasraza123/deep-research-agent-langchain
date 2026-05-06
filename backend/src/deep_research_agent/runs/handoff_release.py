"""Auditable repository-level release manifests for handoff-ready runs."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .handoff_registry import HandoffRegistry, HandoffRegistryItem, read_handoff_registry

ReleaseReadiness = Literal["ready_for_release", "needs_attention", "blocked"]

HANDOFF_RELEASES_DIR = "_handoff/releases"
HANDOFF_RELEASE_JSON = "handoff_release.json"
HANDOFF_RELEASE_MD = "handoff_release.md"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class HandoffReleaseRequest(BaseModel):
    release_id: str | None = None
    requested_by: str = "operator"
    recipient: str = ""
    purpose: str = "external_handoff_release"
    thread_ids: list[str] = Field(default_factory=list)
    include_ready_runs_when_empty: bool = True
    require_registry_ready: bool = True
    require_handoff_ready: bool = True
    require_export_bundle: bool = True
    require_no_missing_controls: bool = True
    overwrite_existing: bool = False
    notes: str = ""


class HandoffReleaseRun(BaseModel):
    thread_id: str
    readiness: str
    handoff_readiness: str
    run_status: str
    review_status: str
    export_present: bool
    export_profile: str = "missing"
    export_archive_sha256: str | None = None
    export_archive_valid: bool | None = None
    custody_readiness: str = "missing"
    integrity_readiness: str = "missing"
    disclosure_readiness: str = "missing"
    disclosure_risk_level: str = "unknown"
    retention_class: str = "missing"
    legal_hold: bool = False
    active_hold_ids: list[str] = Field(default_factory=list)
    operator_audit_valid: bool = True
    missing_controls: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class HandoffReleaseSummary(BaseModel):
    selected_runs: int = 0
    ready_runs: int = 0
    needs_attention_runs: int = 0
    blocked_runs: int = 0
    missing_requested_runs: int = 0
    export_ready_runs: int = 0
    legal_hold_runs: int = 0
    blocker_count: int = 0
    warning_count: int = 0


class HandoffReleaseManifest(BaseModel):
    release_version: str = "1.0"
    release_id: str
    generated_at: str
    requested_by: str = "operator"
    recipient: str = ""
    purpose: str = "external_handoff_release"
    readiness: ReleaseReadiness = "needs_attention"
    registry_generated_at: str
    registry_sha256: str | None = None
    requested_thread_ids: list[str] = Field(default_factory=list)
    missing_thread_ids: list[str] = Field(default_factory=list)
    runs: list[HandoffReleaseRun] = Field(default_factory=list)
    summary: HandoffReleaseSummary = Field(default_factory=HandoffReleaseSummary)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


def build_handoff_release_manifest(
    *,
    runs_dir: Path,
    request: HandoffReleaseRequest | None = None,
    registry: HandoffRegistry | None = None,
) -> HandoffReleaseManifest:
    request = request or HandoffReleaseRequest()
    registry = registry or read_handoff_registry(runs_dir)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    release_id = _release_id(request.release_id)
    release_path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_JSON
    if release_path.exists() and not request.overwrite_existing:
        raise ValueError(f"Handoff release already exists: {release_id}")
    selected_items, missing_thread_ids = _select_registry_items(registry, request)
    if not selected_items:
        raise ValueError("No runs were selected for handoff release.")

    release_runs = [
        _release_run(
            item,
            require_registry_ready=request.require_registry_ready,
            require_handoff_ready=request.require_handoff_ready,
            require_export_bundle=request.require_export_bundle,
            require_no_missing_controls=request.require_no_missing_controls,
        )
        for item in selected_items
    ]
    blockers = _release_blockers(
        release_runs,
        missing_thread_ids,
        request=request,
    )
    warnings = _release_warnings(release_runs, registry)
    readiness: ReleaseReadiness = "ready_for_release"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "needs_attention"

    manifest = HandoffReleaseManifest(
        release_id=release_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        recipient=request.recipient,
        purpose=request.purpose,
        readiness=readiness,
        registry_generated_at=registry.generated_at,
        registry_sha256=_registry_sha256(runs_dir),
        requested_thread_ids=list(request.thread_ids),
        missing_thread_ids=missing_thread_ids,
        runs=release_runs,
        summary=HandoffReleaseSummary(
            selected_runs=len(release_runs),
            ready_runs=sum(1 for item in release_runs if item.readiness == "ready_for_handoff"),
            needs_attention_runs=sum(
                1 for item in release_runs if item.readiness == "needs_attention"
            ),
            blocked_runs=sum(1 for item in release_runs if item.readiness == "blocked"),
            missing_requested_runs=len(missing_thread_ids),
            export_ready_runs=sum(
                1
                for item in release_runs
                if item.export_present and item.export_archive_valid is not False
            ),
            legal_hold_runs=sum(1 for item in release_runs if item.legal_hold),
            blocker_count=len(blockers),
            warning_count=len(warnings),
        ),
        blockers=blockers,
        warnings=warnings,
        required_controls={
            "registry_ready": request.require_registry_ready,
            "handoff_ready": request.require_handoff_ready,
            "export_bundle": request.require_export_bundle,
            "no_missing_controls": request.require_no_missing_controls,
        },
        artifacts=[
            _release_rel_path(release_id, HANDOFF_RELEASE_JSON),
            _release_rel_path(release_id, HANDOFF_RELEASE_MD),
        ],
        notes=request.notes,
    )
    write_handoff_release_manifest(runs_dir, manifest)
    return manifest


def read_handoff_release_manifest(runs_dir: Path, release_id: str) -> HandoffReleaseManifest:
    path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseManifest, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseManifest.parse_obj(data)


def list_handoff_release_manifests(runs_dir: Path) -> list[HandoffReleaseManifest]:
    releases_root = _releases_root(runs_dir)
    if not releases_root.exists():
        return []
    manifests: list[HandoffReleaseManifest] = []
    for path in sorted(releases_root.glob(f"*/{HANDOFF_RELEASE_JSON}")):
        try:
            manifests.append(read_handoff_release_manifest(runs_dir, path.parent.name))
        except Exception:
            continue
    manifests.sort(key=lambda item: item.generated_at, reverse=True)
    return manifests


def write_handoff_release_manifest(
    runs_dir: Path,
    manifest: HandoffReleaseManifest,
) -> list[str]:
    release_dir = _release_dir(runs_dir, manifest.release_id)
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / HANDOFF_RELEASE_JSON).write_text(
        json.dumps(_model_to_plain(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / HANDOFF_RELEASE_MD).write_text(
        render_handoff_release_markdown(manifest),
        encoding="utf-8",
    )
    return [
        _release_rel_path(manifest.release_id, HANDOFF_RELEASE_JSON),
        _release_rel_path(manifest.release_id, HANDOFF_RELEASE_MD),
    ]


def render_handoff_release_markdown(manifest: HandoffReleaseManifest) -> str:
    lines = [
        "# Handoff Release Manifest",
        "",
        f"- Release ID: `{manifest.release_id}`",
        f"- Generated at: `{manifest.generated_at}`",
        f"- Requested by: `{manifest.requested_by}`",
        f"- Recipient: {manifest.recipient or 'unspecified'}",
        f"- Purpose: `{manifest.purpose}`",
        f"- Readiness: `{manifest.readiness}`",
        f"- Registry generated at: `{manifest.registry_generated_at}`",
        f"- Registry SHA-256: `{manifest.registry_sha256 or 'unavailable'}`",
        f"- Selected runs: {manifest.summary.selected_runs}",
        f"- Ready runs: {manifest.summary.ready_runs}",
        f"- Blocked runs: {manifest.summary.blocked_runs}",
        f"- Warnings: {manifest.summary.warning_count}",
        "",
        "## Runs",
        "",
    ]
    for run in manifest.runs:
        lines.extend(
            [
                f"### {run.thread_id}",
                "",
                f"- Registry readiness: `{run.readiness}`",
                f"- Handoff readiness: `{run.handoff_readiness}`",
                f"- Run status: `{run.run_status}`",
                f"- Review status: `{run.review_status}`",
                f"- Export: `{run.export_profile}`; hash valid: `{run.export_archive_valid}`",
                f"- Custody: `{run.custody_readiness}`",
                f"- Integrity: `{run.integrity_readiness}`",
                f"- Disclosure: `{run.disclosure_readiness}` risk `{run.disclosure_risk_level}`",
                f"- Retention: `{run.retention_class}`; legal hold: `{run.legal_hold}`",
                f"- Operator audit valid: `{run.operator_audit_valid}`",
                "- Missing controls: "
                + (", ".join(f"`{control}`" for control in run.missing_controls) or "none"),
                "- Blockers: " + (", ".join(run.blockers) or "none"),
                "- Warnings: " + (", ".join(run.warnings) or "none"),
                "",
            ]
        )
    if manifest.missing_thread_ids:
        lines.extend(["## Missing Requested Runs", ""])
        lines.extend(f"- `{thread_id}`" for thread_id in manifest.missing_thread_ids)
        lines.append("")
    if manifest.blockers:
        lines.extend(["## Release Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in manifest.blockers)
        lines.append("")
    if manifest.warnings:
        lines.extend(["## Release Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest.warnings)
        lines.append("")
    if manifest.notes:
        lines.extend(["## Notes", "", manifest.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _select_registry_items(
    registry: HandoffRegistry,
    request: HandoffReleaseRequest,
) -> tuple[list[HandoffRegistryItem], list[str]]:
    by_thread_id = {item.thread_id: item for item in registry.items}
    if request.thread_ids:
        selected: list[HandoffRegistryItem] = []
        missing: list[str] = []
        seen: set[str] = set()
        for thread_id in request.thread_ids:
            if thread_id in seen:
                continue
            seen.add(thread_id)
            item = by_thread_id.get(thread_id)
            if item is None:
                missing.append(thread_id)
                continue
            selected.append(item)
        return selected, missing
    if not request.include_ready_runs_when_empty:
        return [], []
    return [item for item in registry.items if item.readiness == "ready_for_handoff"], []


def _release_run(
    item: HandoffRegistryItem,
    *,
    require_registry_ready: bool,
    require_handoff_ready: bool,
    require_export_bundle: bool,
    require_no_missing_controls: bool,
) -> HandoffReleaseRun:
    blockers = list(item.blockers)
    warnings = list(item.warnings)
    if require_registry_ready and item.readiness != "ready_for_handoff":
        blockers.append(f"Registry readiness is `{item.readiness}`.")
    if require_handoff_ready and item.handoff_readiness != "ready_for_handoff":
        blockers.append(f"Handoff readiness is `{item.handoff_readiness}`.")
    if require_export_bundle and not item.export_present:
        blockers.append("Export bundle is missing.")
    if require_export_bundle and item.export_archive_valid is False:
        blockers.append("Export archive hash is invalid.")
    if require_no_missing_controls and item.missing_controls:
        blockers.append("Required controls are missing: " + ", ".join(item.missing_controls))
    return HandoffReleaseRun(
        thread_id=item.thread_id,
        readiness=item.readiness,
        handoff_readiness=item.handoff_readiness,
        run_status=item.run_status,
        review_status=item.review_status,
        export_present=item.export_present,
        export_profile=item.export_profile,
        export_archive_sha256=item.export_archive_sha256,
        export_archive_valid=item.export_archive_valid,
        custody_readiness=item.custody_readiness,
        integrity_readiness=item.integrity_readiness,
        disclosure_readiness=item.disclosure_readiness,
        disclosure_risk_level=item.disclosure_risk_level,
        retention_class=item.retention_class,
        legal_hold=item.legal_hold,
        active_hold_ids=list(item.active_hold_ids),
        operator_audit_valid=item.operator_audit_valid,
        missing_controls=list(item.missing_controls),
        blockers=_dedupe(blockers),
        warnings=_dedupe(warnings),
        artifacts=list(item.artifacts),
    )


def _release_blockers(
    runs: list[HandoffReleaseRun],
    missing_thread_ids: list[str],
    *,
    request: HandoffReleaseRequest,
) -> list[str]:
    blockers: list[str] = []
    for thread_id in missing_thread_ids:
        blockers.append(f"Requested run `{thread_id}` is not present in the handoff registry.")
    for run in runs:
        for blocker in run.blockers:
            blockers.append(f"{run.thread_id}: {blocker}")
    if request.require_registry_ready and any(run.readiness != "ready_for_handoff" for run in runs):
        blockers.append("One or more selected runs are not registry-ready.")
    return _dedupe(blockers)


def _release_warnings(
    runs: list[HandoffReleaseRun],
    registry: HandoffRegistry,
) -> list[str]:
    warnings: list[str] = []
    if registry.warnings:
        warnings.extend(f"Registry warning: {warning}" for warning in registry.warnings)
    for run in runs:
        for warning in run.warnings:
            warnings.append(f"{run.thread_id}: {warning}")
        if run.legal_hold:
            holds = ", ".join(run.active_hold_ids) or "policy legal_hold"
            warnings.append(f"{run.thread_id}: legal hold is active ({holds}).")
    return _dedupe(warnings)


def _registry_sha256(runs_dir: Path) -> str | None:
    path = runs_dir / "_handoff" / "handoff_registry.json"
    if not path.exists() or path.is_dir():
        return None
    return file_sha256(path)


def _release_id(value: str | None) -> str:
    release_id = (value or f"release-{uuid.uuid4().hex[:12]}").strip()
    if not RELEASE_ID_PATTERN.match(release_id):
        raise ValueError(
            "release_id must start with an alphanumeric character and contain only "
            "letters, digits, dots, underscores, or hyphens."
        )
    return release_id


def _releases_root(runs_dir: Path) -> Path:
    root = runs_dir.resolve()
    releases_root = (root / HANDOFF_RELEASES_DIR).resolve()
    if root != releases_root and root not in releases_root.parents:
        raise ValueError("Invalid handoff releases directory")
    return releases_root


def _release_dir(runs_dir: Path, release_id: str) -> Path:
    release_id = _release_id(release_id)
    releases_root = _releases_root(runs_dir)
    release_dir = (releases_root / release_id).resolve()
    if releases_root != release_dir and releases_root not in release_dir.parents:
        raise ValueError("Invalid release_id")
    return release_dir


def _release_rel_path(release_id: str, filename: str) -> str:
    return f"{HANDOFF_RELEASES_DIR}/{release_id}/{filename}"


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
