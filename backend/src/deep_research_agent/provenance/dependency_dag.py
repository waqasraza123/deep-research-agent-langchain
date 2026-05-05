from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc

from .contracts import ArtifactDependencyGraph, ArtifactManifest


def build_dependency_dag(manifest: ArtifactManifest) -> ArtifactDependencyGraph:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def node(node_id: str, label: str, kind: str, **extra: Any) -> None:
        nodes.setdefault(node_id, {"id": node_id, "label": label, "kind": kind, **extra})

    def edge(source: str, target: str, relationship: str) -> None:
        edges.append({"source": source, "target": target, "relationship": relationship})

    node("input:question", "Input question and URLs", "input")
    node("settings:run", "Run settings", "settings", fingerprint=manifest.settings_fingerprint)

    for source in manifest.sources:
        source_id = f"source:{source.source_id}"
        node(
            source_id,
            source.url or source.source_id,
            "source",
            content_hash=source.content_hash,
            local_path=source.local_path,
        )
        edge("input:question", source_id, "requested_or_discovered")

    for model in manifest.model_invocations:
        model_id = f"model:{model.provider}:{model.model_name}:{model.purpose}"
        node(
            model_id,
            f"{model.provider}:{model.model_name}",
            "model",
            config_hash=model.config_hash,
            purpose=model.purpose,
        )
        edge("settings:run", model_id, "configured")

    for artifact in manifest.artifacts:
        artifact_id = f"artifact:{artifact.artifact_path}"
        node(
            artifact_id,
            artifact.artifact_path,
            "artifact",
            artifact_type=artifact.artifact_type,
            producer=artifact.producer_subsystem,
            content_hash=artifact.content_hash,
        )
        edge("settings:run", artifact_id, "configured")
        for dep in artifact.input_dependencies:
            dep_id = f"input:{dep.identifier.replace('input.', '')}"
            node(dep_id, dep.identifier, "input")
            edge(dep_id, artifact_id, dep.relationship or "input_to_artifact")
        for dep in artifact.source_dependencies:
            dep_id = f"source:{dep.identifier}"
            if dep_id in nodes:
                edge(dep_id, artifact_id, dep.relationship or "source_to_artifact")
        for dep in artifact.model_dependencies:
            dep_id = f"model:{dep.identifier}"
            if dep_id in nodes:
                edge(dep_id, artifact_id, dep.relationship or "model_to_artifact")
        for dep in artifact.artifact_dependencies:
            if dep.path:
                dep_id = f"artifact:{dep.path}"
                if dep_id in nodes:
                    edge(dep_id, artifact_id, dep.relationship or "artifact_to_artifact")

    _add_canonical_stage_edges(nodes, edges)
    unique_edges = sorted(
        {(_edge["source"], _edge["target"], _edge["relationship"]) for _edge in edges}
    )
    return ArtifactDependencyGraph(
        thread_id=manifest.thread_id,
        generated_at=now_iso_utc(),
        nodes=sorted(nodes.values(), key=lambda item: item["id"]),
        edges=[
            {"source": source, "target": target, "relationship": relationship}
            for source, target, relationship in unique_edges
        ],
    )


def write_dependency_dag(thread_dir: Path, graph: ArtifactDependencyGraph) -> list[str]:
    payload = _model_to_plain(graph)
    (thread_dir / "artifact_dependency_dag.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "artifact_dependency_dag.md").write_text(
        render_dependency_dag_markdown(graph),
        encoding="utf-8",
    )
    return ["artifact_dependency_dag.json", "artifact_dependency_dag.md"]


def render_dependency_dag_markdown(graph: ArtifactDependencyGraph) -> str:
    lines = [
        "# Artifact Dependency DAG",
        "",
        f"- Thread ID: `{graph.thread_id}`",
        f"- Generated at: `{graph.generated_at}`",
        f"- Nodes: {len(graph.nodes)}",
        f"- Edges: {len(graph.edges)}",
        "",
        "## Edges",
        "",
    ]
    if not graph.edges:
        lines.append("- None inferred.")
    for edge in graph.edges:
        lines.append(
            f"- `{edge['source']}` -> `{edge['target']}` ({edge['relationship']})"
        )
    return "\n".join(lines).rstrip() + "\n"


def _add_canonical_stage_edges(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
) -> None:
    stage_edges = [
        ("input:question", "artifact:protocol_selection.json", "produced_protocol_selection"),
        (
            "artifact:source_selection.json",
            "artifact:source_candidates.json",
            "selected_candidates",
        ),
        ("artifact:source_candidates.json", "artifact:sources.json", "source_fetching"),
        ("artifact:sources.json", "artifact:document_chunks.json", "document_intelligence"),
        ("artifact:document_chunks.json", "artifact:context_packs.json", "retrieval"),
        ("artifact:context_packs.json", "artifact:report.md", "agent_context"),
        ("artifact:report.md", "artifact:verification_report.json", "verification"),
        ("artifact:verification_report.json", "artifact:quality_score.json", "evaluation"),
    ]
    for source, target, relationship in stage_edges:
        if source in nodes and target in nodes:
            edges.append({"source": source, "target": target, "relationship": relationship})


def _model_to_plain(model: Any) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
