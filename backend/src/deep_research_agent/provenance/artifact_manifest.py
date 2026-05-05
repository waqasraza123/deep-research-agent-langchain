from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc

from .contracts import ArtifactDependency, ArtifactManifest, ArtifactMetadata
from .lineage import (
    PROVENANCE_ARTIFACTS,
    SUBSYSTEM_VERSION,
    artifact_type_for_path,
    build_model_fingerprints,
    build_run_input_fingerprint,
    build_source_fingerprints,
    build_subsystem_invocations,
    file_sha256,
    infer_artifact_dependencies,
    iter_artifact_files,
    producer_for_path,
    read_events,
    redact_secrets,
    settings_snapshot_from_run,
    stable_hash,
    utc_timestamp_for_file,
)


def build_artifact_manifest(
    thread_dir: Path,
    thread_id: str,
    run: Any | None = None,
) -> ArtifactManifest:
    settings_snapshot = settings_snapshot_from_run(run)
    redacted_settings = redact_secrets(settings_snapshot)
    settings_fingerprint = stable_hash(redacted_settings)
    sources = build_source_fingerprints(thread_dir)
    model_invocations = build_model_fingerprints(redacted_settings, read_events(thread_dir))
    source_dependencies = [
        ArtifactDependency(
            dependency_type="source",
            identifier=source.source_id,
            path=source.local_path,
            content_hash=source.content_hash,
            relationship="grounded_by",
        )
        for source in sources
    ]
    model_dependencies = [
        ArtifactDependency(
            dependency_type="model",
            identifier=f"{model.provider}:{model.model_name}:{model.purpose}",
            content_hash=model.config_hash,
            relationship="generated_or_ranked_by",
        )
        for model in model_invocations
    ]
    input_dependency = ArtifactDependency(
        dependency_type="input",
        identifier="input.question_and_urls",
        content_hash=build_run_input_fingerprint(run).combined_hash
        if build_run_input_fingerprint(run)
        else None,
        relationship="seeded",
    )

    artifacts: list[ArtifactMetadata] = []
    for rel_path, path in iter_artifact_files(thread_dir):
        producer = producer_for_path(rel_path)
        artifact_deps = infer_artifact_dependencies(rel_path)
        artifact_sources = source_dependencies if _uses_sources(rel_path) else []
        artifact_models = model_dependencies if _uses_model(rel_path, producer) else []
        artifact_inputs = [input_dependency] if _uses_input(rel_path, producer) else []
        warnings: list[str] = []
        content_hash = "self-referential" if rel_path in PROVENANCE_ARTIFACTS else file_sha256(path)
        if producer == "unknown":
            warnings.append("Producer subsystem was inferred as unknown.")
        if rel_path in PROVENANCE_ARTIFACTS:
            warnings.append("Content hash is self-referential and intentionally not hashed.")
        artifacts.append(
            ArtifactMetadata(
                artifact_name=Path(rel_path).name,
                artifact_path=rel_path,
                artifact_type=artifact_type_for_path(rel_path),
                created_at=utc_timestamp_for_file(path, "created"),
                updated_at=utc_timestamp_for_file(path, "updated"),
                content_hash=content_hash,
                size_bytes=path.stat().st_size,
                producer_subsystem=producer,
                producer_version=SUBSYSTEM_VERSION,
                input_dependencies=artifact_inputs,
                source_dependencies=artifact_sources,
                model_dependencies=artifact_models,
                artifact_dependencies=artifact_deps,
                settings_fingerprint=settings_fingerprint,
                warnings=warnings,
            )
        )

    return ArtifactManifest(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        run_input=build_run_input_fingerprint(run),
        settings_fingerprint=settings_fingerprint,
        artifacts=artifacts,
        sources=sources,
        model_invocations=model_invocations,
        subsystem_invocations=build_subsystem_invocations(
            [item.artifact_path for item in artifacts]
        ),
        warnings=[],
    )


def write_artifact_manifest(thread_dir: Path, manifest: ArtifactManifest) -> list[str]:
    payload = _model_to_plain(manifest)
    (thread_dir / "artifact_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "artifact_manifest.md").write_text(
        render_artifact_manifest_markdown(manifest),
        encoding="utf-8",
    )
    return ["artifact_manifest.json", "artifact_manifest.md"]


def render_artifact_manifest_markdown(manifest: ArtifactManifest) -> str:
    lines = [
        "# Artifact Manifest",
        "",
        f"- Thread ID: `{manifest.thread_id}`",
        f"- Generated at: `{manifest.generated_at}`",
        f"- Settings fingerprint: `{manifest.settings_fingerprint}`",
        f"- Artifact count: {len(manifest.artifacts)}",
        f"- Source count: {len(manifest.sources)}",
        f"- Model invocation fingerprints: {len(manifest.model_invocations)}",
        "",
        "## Artifacts",
        "",
        "| Artifact | Type | Producer | Size | Hash | Dependencies |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in manifest.artifacts:
        deps = sorted(
            {
                dep.path or dep.identifier
                for dep in [
                    *item.input_dependencies,
                    *item.source_dependencies,
                    *item.model_dependencies,
                    *item.artifact_dependencies,
                ]
            }
        )
        short_hash = (
            item.content_hash[:12]
            if item.content_hash != "self-referential"
            else item.content_hash
        )
        lines.append(
            f"| `{item.artifact_path}` | {item.artifact_type} | {item.producer_subsystem} | "
            f"{item.size_bytes} | `{short_hash}` | {', '.join(deps) or '-'} |"
        )
    lines.extend(["", "## Sources", ""])
    if not manifest.sources:
        lines.append("- None captured.")
    for source in manifest.sources:
        hash_label = source.content_hash[:12] if source.content_hash else "missing"
        lines.append(f"- `{source.source_id}` {source.url} hash=`{hash_label}`")
    lines.extend(["", "## Model Fingerprints", ""])
    if not manifest.model_invocations:
        lines.append("- None captured.")
    for model in manifest.model_invocations:
        nondeterminism = "nondeterministic" if model.nondeterministic else "deterministic"
        lines.append(
            f"- {model.provider}:{model.model_name} purpose=`{model.purpose}` "
            f"config_hash=`{model.config_hash[:12]}` {nondeterminism}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _uses_sources(path: str) -> bool:
    name = Path(path).name
    return name in {"report.md", "notes.md"} or any(
        marker in name
        for marker in (
            "document",
            "retrieval",
            "context",
            "evidence",
            "verification",
            "evaluation",
            "quality",
            "source_audit",
            "citation",
            "synthesis",
        )
    )


def _uses_model(path: str, producer: str) -> bool:
    if producer == "provenance":
        return False
    return producer in {
        "agent",
        "retrieval",
        "verification",
        "synthesis",
        "evaluation",
        "protocol",
        "source_discovery",
    }


def _uses_input(path: str, producer: str) -> bool:
    if producer == "provenance":
        return True
    return producer != "runtime" or Path(path).name == "run.json"


def _model_to_plain(model: Any) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
