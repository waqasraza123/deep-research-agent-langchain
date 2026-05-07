"""Portable transfer bundles for handoff release manifests."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.provenance.lineage import file_sha256

from .export_bundle import export_bundle_path
from .handoff_release import (
    HANDOFF_RELEASE_JSON,
    HANDOFF_RELEASE_MD,
    HANDOFF_RELEASES_DIR,
    read_handoff_release_manifest,
)
from .handoff_release_verification import (
    HANDOFF_RELEASE_VERIFICATION_JSON,
    HANDOFF_RELEASE_VERIFICATION_MD,
    read_handoff_release_verification_report,
)

ReleaseBundleReadiness = Literal["ready", "warnings", "blocked"]
ReleaseBundleEntryType = Literal[
    "release_artifact",
    "verification_artifact",
    "registry_artifact",
    "run_export",
    "bundle_manifest",
]

HANDOFF_RELEASE_BUNDLE_NAME = "handoff_release_bundle.zip"
HANDOFF_RELEASE_BUNDLE_JSON = "handoff_release_bundle_manifest.json"
HANDOFF_RELEASE_BUNDLE_MD = "handoff_release_bundle_manifest.md"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class HandoffReleaseBundleRequest(BaseModel):
    requested_by: str = "operator"
    require_release_ready: bool = True
    require_verification_valid: bool = True
    include_verification: bool = True
    include_registry_snapshot: bool = True
    include_run_exports: bool = True
    require_run_exports: bool = True
    overwrite_existing: bool = True
    max_total_input_bytes: int = Field(default=2_000_000_000, ge=1)
    notes: str = ""


class HandoffReleaseBundleEntry(BaseModel):
    path: str
    archive_path: str
    entry_type: ReleaseBundleEntryType
    size_bytes: int
    sha256: str


class HandoffReleaseBundleRun(BaseModel):
    thread_id: str
    export_included: bool = False
    export_archive_path: str | None = None
    expected_export_sha256: str | None = None
    actual_export_sha256: str | None = None
    export_status: Literal["included", "missing", "hash_mismatch", "skipped"] = "skipped"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HandoffReleaseBundleSummary(BaseModel):
    entries: int = 0
    selected_runs: int = 0
    included_run_exports: int = 0
    missing_run_exports: int = 0
    hash_mismatched_run_exports: int = 0
    total_input_bytes: int = 0
    blocker_count: int = 0
    warning_count: int = 0


class HandoffReleaseBundleManifest(BaseModel):
    bundle_version: str = "1.0"
    release_id: str
    generated_at: str
    requested_by: str = "operator"
    readiness: ReleaseBundleReadiness = "warnings"
    release_readiness: str
    release_generated_at: str
    verification_readiness: str = "missing"
    archive_path: str
    archive_size_bytes: int = 0
    archive_sha256: str = ""
    release_sha256: str | None = None
    verification_sha256: str | None = None
    entries: list[HandoffReleaseBundleEntry] = Field(default_factory=list)
    run_exports: list[HandoffReleaseBundleRun] = Field(default_factory=list)
    summary: HandoffReleaseBundleSummary = Field(default_factory=HandoffReleaseBundleSummary)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_controls: dict[str, bool] = Field(default_factory=dict)
    notes: str = ""


def build_handoff_release_bundle(
    *,
    runs_dir: Path,
    release_id: str,
    request: HandoffReleaseBundleRequest | None = None,
) -> HandoffReleaseBundleManifest:
    request = request or HandoffReleaseBundleRequest()
    release = read_handoff_release_manifest(runs_dir, release_id)
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    release_dir = _release_dir(runs_dir, release.release_id)
    archive_path = release_dir / HANDOFF_RELEASE_BUNDLE_NAME
    if archive_path.exists() and not request.overwrite_existing:
        raise ValueError(f"Handoff release bundle already exists: {release.release_id}")

    blockers: list[str] = []
    warnings: list[str] = []
    if request.require_release_ready and release.readiness != "ready_for_release":
        blockers.append(f"Release readiness is `{release.readiness}`.")

    verification_readiness = "missing"
    verification_sha256: str | None = None
    try:
        verification = read_handoff_release_verification_report(runs_dir, release.release_id)
    except FileNotFoundError:
        verification = None
        if request.require_verification_valid:
            blockers.append("Handoff release verification report is missing.")
        elif request.include_verification:
            warnings.append("Handoff release verification report is missing.")
    else:
        verification_readiness = verification.readiness
        verification_sha256 = _file_sha256_or_none(
            release_dir / HANDOFF_RELEASE_VERIFICATION_JSON
        )
        if request.require_verification_valid and verification.readiness != "valid":
            blockers.append(f"Verification readiness is `{verification.readiness}`.")

    entries: list[HandoffReleaseBundleEntry] = []
    run_exports: list[HandoffReleaseBundleRun] = []
    total_input_bytes = 0

    candidate_files: list[tuple[Path, str, ReleaseBundleEntryType]] = [
        (
            release_dir / HANDOFF_RELEASE_JSON,
            f"release/{HANDOFF_RELEASE_JSON}",
            "release_artifact",
        ),
        (
            release_dir / HANDOFF_RELEASE_MD,
            f"release/{HANDOFF_RELEASE_MD}",
            "release_artifact",
        ),
    ]
    if request.include_verification and verification is not None:
        candidate_files.extend(
            [
                (
                    release_dir / HANDOFF_RELEASE_VERIFICATION_JSON,
                    f"release/{HANDOFF_RELEASE_VERIFICATION_JSON}",
                    "verification_artifact",
                ),
                (
                    release_dir / HANDOFF_RELEASE_VERIFICATION_MD,
                    f"release/{HANDOFF_RELEASE_VERIFICATION_MD}",
                    "verification_artifact",
                ),
            ]
        )
    if request.include_registry_snapshot:
        candidate_files.append(
            (
                runs_dir / "_handoff" / "handoff_registry.json",
                "registry/handoff_registry.json",
                "registry_artifact",
            )
        )
        candidate_files.append(
            (
                runs_dir / "_handoff" / "handoff_registry.md",
                "registry/handoff_registry.md",
                "registry_artifact",
            )
        )

    for path, archive_name, entry_type in candidate_files:
        if not path.exists() or path.is_dir():
            message = f"Required bundle source artifact is missing: {path.name}."
            if entry_type == "registry_artifact":
                warnings.append(message)
            else:
                blockers.append(message)
            continue
        entry, total_input_bytes, size_blocker = _bounded_entry(
            runs_dir,
            path,
            archive_name,
            entry_type,
            current_total=total_input_bytes,
            max_total=request.max_total_input_bytes,
        )
        if size_blocker:
            blockers.append(size_blocker)
            continue
        entries.append(entry)

    if request.include_run_exports:
        for release_run in release.runs:
            run_export = _run_export_entry(
                runs_dir,
                release_run.thread_id,
                expected_sha256=release_run.export_archive_sha256,
                require_run_exports=request.require_run_exports,
            )
            run_exports.append(run_export)
            blockers.extend(f"{run_export.thread_id}: {item}" for item in run_export.blockers)
            warnings.extend(f"{run_export.thread_id}: {item}" for item in run_export.warnings)
            if run_export.export_archive_path:
                export_path = export_bundle_path(runs_dir, release_run.thread_id)
                entry, total_input_bytes, size_blocker = _bounded_entry(
                    runs_dir,
                    export_path,
                    run_export.export_archive_path,
                    "run_export",
                    current_total=total_input_bytes,
                    max_total=request.max_total_input_bytes,
                )
                if size_blocker:
                    run_export.export_included = False
                    run_export.export_status = "skipped"
                    run_export.export_archive_path = None
                    run_export.blockers.append(size_blocker)
                    blockers.append(f"{release_run.thread_id}: {size_blocker}")
                    continue
                entries.append(entry)

    blockers = _dedupe(blockers)
    warnings = _dedupe(warnings)
    readiness: ReleaseBundleReadiness = "ready"
    if blockers:
        readiness = "blocked"
    elif warnings:
        readiness = "warnings"

    manifest = HandoffReleaseBundleManifest(
        release_id=release.release_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        release_readiness=release.readiness,
        release_generated_at=release.generated_at,
        verification_readiness=verification_readiness,
        archive_path=_release_rel_path(release.release_id, HANDOFF_RELEASE_BUNDLE_NAME),
        release_sha256=_file_sha256_or_none(release_dir / HANDOFF_RELEASE_JSON),
        verification_sha256=verification_sha256,
        entries=entries,
        run_exports=run_exports,
        summary=HandoffReleaseBundleSummary(
            entries=len(entries) + 2,
            selected_runs=len(release.runs),
            included_run_exports=sum(1 for item in run_exports if item.export_included),
            missing_run_exports=sum(1 for item in run_exports if item.export_status == "missing"),
            hash_mismatched_run_exports=sum(
                1 for item in run_exports if item.export_status == "hash_mismatch"
            ),
            total_input_bytes=total_input_bytes,
            blocker_count=len(blockers),
            warning_count=len(warnings),
        ),
        blockers=blockers,
        warnings=warnings,
        required_controls={
            "release_ready": request.require_release_ready,
            "verification_valid": request.require_verification_valid,
            "run_exports": request.require_run_exports,
        },
        notes=request.notes,
    )
    release_dir.mkdir(parents=True, exist_ok=True)
    _write_bundle_archive(runs_dir, archive_path, manifest)
    manifest.archive_size_bytes = archive_path.stat().st_size
    manifest.archive_sha256 = file_sha256(archive_path)
    write_handoff_release_bundle_manifest(runs_dir, manifest)
    return manifest


def read_handoff_release_bundle_manifest(
    runs_dir: Path,
    release_id: str,
) -> HandoffReleaseBundleManifest:
    path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_BUNDLE_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(HandoffReleaseBundleManifest, "model_validate", None)
    if callable(validate):
        return validate(data)
    return HandoffReleaseBundleManifest.parse_obj(data)


def handoff_release_bundle_path(runs_dir: Path, release_id: str) -> Path:
    path = _release_dir(runs_dir, release_id) / HANDOFF_RELEASE_BUNDLE_NAME
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    return path


def write_handoff_release_bundle_manifest(
    runs_dir: Path,
    manifest: HandoffReleaseBundleManifest,
) -> list[str]:
    release_dir = _release_dir(runs_dir, manifest.release_id)
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / HANDOFF_RELEASE_BUNDLE_JSON).write_text(
        json.dumps(_model_to_plain(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / HANDOFF_RELEASE_BUNDLE_MD).write_text(
        render_handoff_release_bundle_markdown(manifest),
        encoding="utf-8",
    )
    return [
        _release_rel_path(manifest.release_id, HANDOFF_RELEASE_BUNDLE_NAME),
        _release_rel_path(manifest.release_id, HANDOFF_RELEASE_BUNDLE_JSON),
        _release_rel_path(manifest.release_id, HANDOFF_RELEASE_BUNDLE_MD),
    ]


def render_handoff_release_bundle_markdown(
    manifest: HandoffReleaseBundleManifest,
) -> str:
    lines = [
        "# Handoff Release Bundle",
        "",
        f"- Release ID: `{manifest.release_id}`",
        f"- Generated at: `{manifest.generated_at}`",
        f"- Requested by: `{manifest.requested_by}`",
        f"- Readiness: `{manifest.readiness}`",
        f"- Release readiness: `{manifest.release_readiness}`",
        f"- Verification readiness: `{manifest.verification_readiness}`",
        f"- Archive: `{manifest.archive_path}`",
        f"- Archive SHA-256: `{manifest.archive_sha256 or 'pending'}`",
        f"- Archive size: {manifest.archive_size_bytes} bytes",
        f"- Entries: {manifest.summary.entries}",
        f"- Selected runs: {manifest.summary.selected_runs}",
        f"- Included run exports: {manifest.summary.included_run_exports}",
        f"- Missing run exports: {manifest.summary.missing_run_exports}",
        f"- Export hash mismatches: {manifest.summary.hash_mismatched_run_exports}",
        "",
        "## Bundle Entries",
        "",
    ]
    if manifest.entries:
        for entry in manifest.entries:
            lines.append(
                f"- `{entry.archive_path}` from `{entry.path}` "
                f"({entry.entry_type}, {entry.size_bytes} bytes, sha256=`{entry.sha256}`)"
            )
    else:
        lines.append("- None")
    if manifest.run_exports:
        lines.extend(["", "## Run Exports", ""])
        for run in manifest.run_exports:
            lines.extend(
                [
                    f"### {run.thread_id}",
                    "",
                    f"- Status: `{run.export_status}`",
                    f"- Included: `{run.export_included}`",
                    f"- Archive path: `{run.export_archive_path or 'none'}`",
                    f"- Expected SHA-256: `{run.expected_export_sha256 or 'unavailable'}`",
                    f"- Actual SHA-256: `{run.actual_export_sha256 or 'unavailable'}`",
                    "- Blockers: " + (", ".join(run.blockers) or "none"),
                    "- Warnings: " + (", ".join(run.warnings) or "none"),
                    "",
                ]
            )
    if manifest.blockers:
        lines.extend(["## Bundle Blockers", ""])
        lines.extend(f"- {item}" for item in manifest.blockers)
        lines.append("")
    if manifest.warnings:
        lines.extend(["## Bundle Warnings", ""])
        lines.extend(f"- {item}" for item in manifest.warnings)
        lines.append("")
    if manifest.notes:
        lines.extend(["## Notes", "", manifest.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _write_bundle_archive(
    runs_dir: Path,
    path: Path,
    manifest: HandoffReleaseBundleManifest,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in sorted(manifest.entries, key=lambda item: item.archive_path):
            source = _source_path_for_entry(runs_dir, manifest.release_id, entry)
            if source.exists() and source.is_file():
                _write_zip_bytes(archive, entry.archive_path, source.read_bytes())
        _write_zip_bytes(
            archive,
            HANDOFF_RELEASE_BUNDLE_JSON,
            _json_bytes(_model_to_plain(manifest)),
        )
        _write_zip_bytes(
            archive,
            HANDOFF_RELEASE_BUNDLE_MD,
            render_handoff_release_bundle_markdown(manifest).encode("utf-8"),
        )


def _source_path_for_entry(
    runs_dir: Path,
    release_id: str,
    entry: HandoffReleaseBundleEntry,
) -> Path:
    del release_id
    root = runs_dir.resolve()
    source = (root / entry.path).resolve()
    if root != source and root not in source.parents:
        raise ValueError("Invalid bundle entry path")
    return source


def _run_export_entry(
    runs_dir: Path,
    thread_id: str,
    *,
    expected_sha256: str | None,
    require_run_exports: bool,
) -> HandoffReleaseBundleRun:
    try:
        path = export_bundle_path(runs_dir, thread_id)
    except FileNotFoundError:
        message = "Export archive is missing."
        return HandoffReleaseBundleRun(
            thread_id=thread_id,
            expected_export_sha256=expected_sha256,
            export_status="missing",
            blockers=[message] if require_run_exports else [],
            warnings=[] if require_run_exports else [message],
        )
    actual_sha256 = file_sha256(path)
    if expected_sha256 and expected_sha256 != actual_sha256:
        message = "Export archive hash differs from the release manifest."
        return HandoffReleaseBundleRun(
            thread_id=thread_id,
            expected_export_sha256=expected_sha256,
            actual_export_sha256=actual_sha256,
            export_status="hash_mismatch",
            blockers=[message] if require_run_exports else [],
            warnings=[] if require_run_exports else [message],
        )
    archive_path = f"run_exports/{thread_id}/run_export.zip"
    return HandoffReleaseBundleRun(
        thread_id=thread_id,
        export_included=True,
        export_archive_path=archive_path,
        expected_export_sha256=expected_sha256,
        actual_export_sha256=actual_sha256,
        export_status="included",
    )


def _bounded_entry(
    runs_dir: Path,
    path: Path,
    archive_path: str,
    entry_type: ReleaseBundleEntryType,
    *,
    current_total: int,
    max_total: int,
) -> tuple[HandoffReleaseBundleEntry, int, str | None]:
    size = path.stat().st_size
    next_total = current_total + size
    entry = _entry(runs_dir, path, archive_path, entry_type)
    if next_total > max_total:
        return (
            entry,
            current_total,
            "Bundle input size would exceed max_total_input_bytes "
            f"({next_total} > {max_total}).",
        )
    return entry, next_total, None


def _entry(
    runs_dir: Path,
    path: Path,
    archive_path: str,
    entry_type: ReleaseBundleEntryType,
) -> HandoffReleaseBundleEntry:
    return HandoffReleaseBundleEntry(
        path=_relative_path(runs_dir, path),
        archive_path=archive_path,
        entry_type=entry_type,
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def _write_zip_bytes(archive: zipfile.ZipFile, archive_path: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(archive_path, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def _relative_path(runs_dir: Path, path: Path) -> str:
    root = runs_dir.resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("Bundle source path is outside runs_dir")
    return str(resolved.relative_to(root)).replace("\\", "/")


def _file_sha256_or_none(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    return file_sha256(path)


def _release_dir(runs_dir: Path, release_id: str) -> Path:
    release_id = _safe_release_id(release_id)
    root = runs_dir.resolve()
    release_root = (root / HANDOFF_RELEASES_DIR).resolve()
    release_dir = (release_root / release_id).resolve()
    if release_root != release_dir and release_root not in release_dir.parents:
        raise ValueError("Invalid release_id")
    return release_dir


def _release_rel_path(release_id: str, filename: str) -> str:
    return f"{HANDOFF_RELEASES_DIR}/{release_id}/{filename}"


def _safe_release_id(release_id: str) -> str:
    release_id = release_id.strip()
    if not RELEASE_ID_PATTERN.match(release_id):
        raise ValueError("Invalid release_id")
    return release_id


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


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
