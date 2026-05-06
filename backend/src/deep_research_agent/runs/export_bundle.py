from __future__ import annotations

import fnmatch
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import INTERNAL_FILES, now_iso_utc, safe_thread_id
from deep_research_agent.provenance.lineage import file_sha256, redact_secrets

ExportProfile = Literal["audit", "public", "full"]

EXPORT_DIR = "exports"
EXPORT_BUNDLE_NAME = "run_export.zip"
EXPORT_MANIFEST_JSON = "export_manifest.json"
EXPORT_MANIFEST_MD = "export_manifest.md"

EXPORT_ARTIFACT_PREFIXES = ("exports/",)
RAW_SOURCE_PREFIXES = ("sources/", "sanitized_sources/")

PUBLIC_INCLUDE_PATTERNS = (
    "plan.md",
    "notes.md",
    "report.md",
    "sources.json",
    "run.json",
    "advanced_intelligence_summary.*",
    "intelligence_pipeline_summary.*",
    "intelligence_summary.*",
    "kernel_summary.*",
    "research_readiness.md",
    "artifact_manifest.*",
    "artifact_dependency_dag.*",
    "reproducibility_report.*",
    "replay_plan.*",
    "custody_certificate.*",
    "integrity_report.*",
    "disclosure_report.*",
    "handoff_manifest.*",
    "quality_score.*",
    "evaluation.*",
    "verification_report.*",
    "confidence_calibration.*",
    "evidence_ledger.*",
    "unsupported_claims.md",
    "contradictions.md",
    "source_audit.*",
    "source_safety.*",
    "temporal_profile.*",
    "currentness_assessment.*",
    "quantitative_profile.*",
    "hypotheses.*",
    "decision_memo.*",
    "uncertainty_boundaries.*",
    "replay_execution.*",
)

AUDIT_INCLUDE_PATTERNS = (
    "*",
)

DEFAULT_EXCLUDE_PATTERNS = (
    ".run.json",
    ".cancel.json",
    "*.tmp",
    "*.sqlite",
    "*.db",
    "*.db-shm",
    "*.db-wal",
    "__pycache__/*",
    "exports/*",
)

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".csv",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
}

SECRET_TEXT_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}", re.I),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,'\"]+"),
)


class RunExportRequest(BaseModel):
    profile: ExportProfile = "audit"
    include_raw_sources: bool = False
    include_internal: bool = False
    redact: bool = True
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    max_total_bytes: int = Field(default=100_000_000, ge=1)
    max_file_bytes: int = Field(default=25_000_000, ge=1)
    notes: str = ""


class ExportedArtifact(BaseModel):
    path: str
    archive_path: str
    size_bytes: int
    exported_size_bytes: int
    sha256: str
    exported_sha256: str
    redacted: bool = False
    skipped_reason: str | None = None


class SkippedArtifact(BaseModel):
    path: str
    reason: str
    size_bytes: int | None = None


class RunExportManifest(BaseModel):
    manifest_version: str = "1.0"
    thread_id: str
    generated_at: str
    profile: ExportProfile
    archive_path: str
    archive_size_bytes: int = 0
    archive_sha256: str = ""
    include_raw_sources: bool = False
    include_internal: bool = False
    redact: bool = True
    max_total_bytes: int
    max_file_bytes: int
    exported_count: int = 0
    skipped_count: int = 0
    total_source_bytes: int = 0
    total_exported_bytes: int = 0
    artifacts: list[ExportedArtifact] = Field(default_factory=list)
    skipped_artifacts: list[SkippedArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


def build_run_export_bundle(
    *,
    runs_dir: Path,
    thread_id: str,
    request: RunExportRequest | None = None,
) -> RunExportManifest:
    request = request or RunExportRequest()
    thread_dir = _safe_run_dir(runs_dir, thread_id)
    export_dir = thread_dir / EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_rel = f"{EXPORT_DIR}/{EXPORT_BUNDLE_NAME}"
    archive_path = thread_dir / archive_rel

    candidates = _iter_candidate_files(thread_dir, include_internal=request.include_internal)
    selected, skipped = _select_files(candidates, request)
    exported: list[ExportedArtifact] = []
    warnings: list[str] = []
    total_source_bytes = 0
    total_exported_bytes = 0

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, path in selected:
            size = path.stat().st_size
            total_source_bytes += size
            if size > request.max_file_bytes:
                skipped.append(
                    SkippedArtifact(
                        path=rel_path,
                        reason="file exceeds max_file_bytes",
                        size_bytes=size,
                    )
                )
                continue
            if total_source_bytes > request.max_total_bytes:
                skipped.append(
                    SkippedArtifact(
                        path=rel_path,
                        reason="export exceeds max_total_bytes",
                        size_bytes=size,
                    )
                )
                continue

            raw_hash = file_sha256(path)
            payload, redacted = _export_payload(path, redact=request.redact)
            archive_path_for_file = f"artifacts/{rel_path}"
            archive.writestr(archive_path_for_file, payload)
            exported_hash = _bytes_sha256(payload)
            total_exported_bytes += len(payload)
            exported.append(
                ExportedArtifact(
                    path=rel_path,
                    archive_path=archive_path_for_file,
                    size_bytes=size,
                    exported_size_bytes=len(payload),
                    sha256=raw_hash,
                    exported_sha256=exported_hash,
                    redacted=redacted,
                )
            )

        manifest = RunExportManifest(
            thread_id=thread_id,
            generated_at=now_iso_utc(),
            profile=request.profile,
            archive_path=archive_rel,
            include_raw_sources=request.include_raw_sources,
            include_internal=request.include_internal,
            redact=request.redact,
            max_total_bytes=request.max_total_bytes,
            max_file_bytes=request.max_file_bytes,
            exported_count=len(exported),
            skipped_count=len(skipped),
            total_source_bytes=total_source_bytes,
            total_exported_bytes=total_exported_bytes,
            artifacts=exported,
            skipped_artifacts=skipped,
            warnings=warnings,
            notes=request.notes,
        )
        archive.writestr(EXPORT_MANIFEST_JSON, _json_bytes(_model_to_plain(manifest)))
        archive.writestr(
            EXPORT_MANIFEST_MD,
            render_export_manifest_markdown(manifest).encode("utf-8"),
        )

    manifest.archive_size_bytes = archive_path.stat().st_size
    manifest.archive_sha256 = file_sha256(archive_path)
    manifest.exported_count = len(exported)
    manifest.skipped_count = len(skipped)
    _write_manifest_files(export_dir, manifest)
    return manifest


def read_export_manifest(runs_dir: Path, thread_id: str) -> RunExportManifest:
    thread_dir = _safe_run_dir(runs_dir, thread_id)
    path = thread_dir / EXPORT_DIR / EXPORT_MANIFEST_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(RunExportManifest, "model_validate", None)
    if callable(validate):
        return validate(data)
    return RunExportManifest.parse_obj(data)


def export_bundle_path(runs_dir: Path, thread_id: str) -> Path:
    thread_dir = _safe_run_dir(runs_dir, thread_id)
    path = thread_dir / EXPORT_DIR / EXPORT_BUNDLE_NAME
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    return path


def render_export_manifest_markdown(manifest: RunExportManifest) -> str:
    lines = [
        "# Run Export Manifest",
        "",
        f"- Thread ID: `{manifest.thread_id}`",
        f"- Generated at: `{manifest.generated_at}`",
        f"- Profile: `{manifest.profile}`",
        f"- Archive: `{manifest.archive_path}`",
        f"- Archive SHA-256: `{manifest.archive_sha256 or 'pending'}`",
        f"- Redaction enabled: `{manifest.redact}`",
        f"- Raw source files included: `{manifest.include_raw_sources}`",
        f"- Exported artifacts: {manifest.exported_count}",
        f"- Skipped artifacts: {manifest.skipped_count}",
        f"- Source bytes considered: {manifest.total_source_bytes}",
        f"- Exported bytes before zip compression: {manifest.total_exported_bytes}",
        "",
        "## Exported Artifacts",
        "",
    ]
    if manifest.artifacts:
        for artifact in manifest.artifacts:
            redacted = " redacted" if artifact.redacted else ""
            lines.append(
                f"- `{artifact.path}` -> `{artifact.archive_path}` "
                f"({artifact.exported_size_bytes} bytes,{redacted} "
                f"sha256=`{artifact.exported_sha256}`)"
            )
    else:
        lines.append("- None")
    if manifest.skipped_artifacts:
        lines.extend(["", "## Skipped Artifacts", ""])
        for artifact in manifest.skipped_artifacts:
            size = f", {artifact.size_bytes} bytes" if artifact.size_bytes is not None else ""
            lines.append(f"- `{artifact.path}`: {artifact.reason}{size}")
    if manifest.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest.warnings)
    if manifest.notes:
        lines.extend(["", "## Notes", "", manifest.notes.strip()])
    return "\n".join(lines).rstrip() + "\n"


def _iter_candidate_files(thread_dir: Path, *, include_internal: bool) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    root = thread_dir.resolve()
    for path in thread_dir.rglob("*"):
        if path.is_dir():
            continue
        resolved = path.resolve()
        if root != resolved and root not in resolved.parents:
            continue
        rel_path = str(resolved.relative_to(root)).replace("\\", "/")
        if rel_path in INTERNAL_FILES and not include_internal:
            continue
        if any(rel_path.startswith(prefix) for prefix in EXPORT_ARTIFACT_PREFIXES):
            continue
        out.append((rel_path, resolved))
    out.sort(key=lambda item: item[0])
    return out


def _select_files(
    candidates: list[tuple[str, Path]],
    request: RunExportRequest,
) -> tuple[list[tuple[str, Path]], list[SkippedArtifact]]:
    include_patterns = _include_patterns_for_request(request)
    exclude_patterns = (*DEFAULT_EXCLUDE_PATTERNS, *request.exclude_patterns)
    selected: list[tuple[str, Path]] = []
    skipped: list[SkippedArtifact] = []
    for rel_path, path in candidates:
        size = path.stat().st_size if path.exists() else None
        if not request.include_raw_sources and rel_path.startswith(RAW_SOURCE_PREFIXES):
            skipped.append(
                SkippedArtifact(
                    path=rel_path,
                    reason="raw source files excluded",
                    size_bytes=size,
                )
            )
            continue
        if _matches_any(rel_path, exclude_patterns):
            skipped.append(
                SkippedArtifact(path=rel_path, reason="excluded by pattern", size_bytes=size)
            )
            continue
        if include_patterns and not _matches_any(rel_path, include_patterns):
            skipped.append(
                SkippedArtifact(path=rel_path, reason="not included by profile", size_bytes=size)
            )
            continue
        selected.append((rel_path, path))
    return selected, skipped


def _include_patterns_for_request(request: RunExportRequest) -> tuple[str, ...]:
    if request.include_patterns:
        return tuple(request.include_patterns)
    if request.profile == "full":
        return ("*",)
    if request.profile == "public":
        return PUBLIC_INCLUDE_PATTERNS
    return AUDIT_INCLUDE_PATTERNS


def _matches_any(rel_path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _export_payload(path: Path, *, redact: bool) -> tuple[bytes, bool]:
    if not redact:
        return path.read_bytes(), False
    suffix = path.suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        return path.read_bytes(), False
    text = path.read_text(encoding="utf-8", errors="replace")
    redacted = False
    if suffix in {".json", ".jsonl"}:
        payload, changed = _redact_json_text(text, jsonl=suffix == ".jsonl")
        if changed:
            redacted = True
            return payload.encode("utf-8"), True
    redacted_text = _redact_text(text)
    if redacted_text != text:
        redacted = True
    return redacted_text.encode("utf-8"), redacted


def _redact_json_text(text: str, *, jsonl: bool) -> tuple[str, bool]:
    try:
        if jsonl:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            redacted_rows = [redact_secrets(row) for row in rows]
            payload = "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) for row in redacted_rows
            )
            return payload + "\n", redacted_rows != rows
        loaded = json.loads(text)
    except Exception:
        redacted_text = _redact_text(text)
        return redacted_text, redacted_text != text
    redacted = redact_secrets(loaded)
    return (
        json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        redacted != loaded,
    )


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_TEXT_PATTERNS:
        redacted = pattern.sub(lambda match: _redacted_replacement(match.group(0)), redacted)
    return redacted


def _redacted_replacement(value: str) -> str:
    if ":" in value or "=" in value:
        key = re.split(r"[:=]", value, maxsplit=1)[0]
        return f"{key}=[REDACTED]"
    if value.lower().startswith("bearer "):
        return "Bearer [REDACTED]"
    return "[REDACTED]"


def _write_manifest_files(export_dir: Path, manifest: RunExportManifest) -> None:
    (export_dir / EXPORT_MANIFEST_JSON).write_text(
        json.dumps(_model_to_plain(manifest), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (export_dir / EXPORT_MANIFEST_MD).write_text(
        render_export_manifest_markdown(manifest),
        encoding="utf-8",
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _bytes_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()


def _safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    thread_dir = (root / thread_id).resolve()
    if root != thread_dir and root not in thread_dir.parents:
        raise ValueError("Invalid thread_id")
    if not thread_dir.exists() or not thread_dir.is_dir():
        raise FileNotFoundError(thread_id)
    return thread_dir
