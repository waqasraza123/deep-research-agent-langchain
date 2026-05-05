from __future__ import annotations

import json
from pathlib import Path

from .contracts import MemoryContext, MemoryGraph


def _dump_model(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def render_memory_context_markdown(context: MemoryContext) -> str:
    lines = [
        "# Memory Context",
        "",
        "This is prior context from earlier local runs. It is not fresh evidence.",
        "",
        f"- Confidence: {context.confidence_score:.3f}",
        f"- Similar previous questions: {len(context.similar_previous_questions)}",
        f"- Source reuse candidates: {len(context.suggested_source_reuse_candidates)}",
        "",
        "## Similar Questions",
        "",
    ]
    if not context.similar_previous_questions:
        lines.append("- None")
    for record in context.similar_previous_questions:
        lines.append(f"- `{record.thread_id}`: {record.question}")
        if record.source_title:
            lines.append(f"  - Source: {record.source_title} ({record.source_url})")

    lines.extend(["", "## Suggested Source Reuse", ""])
    if not context.suggested_source_reuse_candidates:
        lines.append("- None")
    for decision in context.suggested_source_reuse_candidates:
        status = "allowed" if decision.reuse_allowed else "suggested only"
        lines.append(
            f"- {status}: {decision.reuse_reason} "
            f"(confidence {decision.confidence_score:.3f})"
        )
        if decision.previous_thread_ids:
            lines.append(f"  - Previous runs: {', '.join(decision.previous_thread_ids)}")
        if decision.freshness_warning:
            lines.append(f"  - Freshness: {decision.freshness_warning}")

    lines.extend(["", "## Known Entities", ""])
    if not context.known_entities:
        lines.append("- None")
    for entity in context.known_entities[:25]:
        lines.append(
            f"- {entity.name} ({entity.entity_type.value}, "
            f"confidence {entity.confidence:.3f}, mentions {entity.mentions})"
        )

    lines.extend(["", "## Known Topics", ""])
    if not context.known_topics:
        lines.append("- None")
    for topic in context.known_topics[:25]:
        lines.append(f"- {topic.name} (score {topic.score:.3f})")

    lines.extend(["", "## Prior Artifacts", ""])
    if not context.prior_artifact_links:
        lines.append("- None")
    for artifact in context.prior_artifact_links[:30]:
        lines.append(f"- `runs/{artifact.thread_id}/{artifact.path}` ({artifact.artifact_type})")

    if context.stale_warnings:
        lines.extend(["", "## Stale Warnings", ""])
        for warning in context.stale_warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).rstrip() + "\n"


def write_memory_context_artifacts(thread_dir: Path, context: MemoryContext) -> None:
    thread_dir.mkdir(parents=True, exist_ok=True)
    (thread_dir / "memory_context.json").write_text(
        json.dumps(_dump_model(context), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "memory_context.md").write_text(
        render_memory_context_markdown(context),
        encoding="utf-8",
    )


def render_memory_graph_markdown(graph: MemoryGraph) -> str:
    counts: dict[str, int] = {}
    for node in graph.nodes:
        counts[node.type] = counts.get(node.type, 0) + 1
    lines = [
        "# Memory Graph",
        "",
        f"- Generated at: `{graph.generated_at}`",
        f"- Nodes: {len(graph.nodes)}",
        f"- Edges: {len(graph.edges)}",
        "",
        "## Node Counts",
        "",
    ]
    for node_type, count in sorted(counts.items()):
        lines.append(f"- {node_type}: {count}")

    lines.extend(["", "## Reuse Relationships", ""])
    reuse_edges = [edge for edge in graph.edges if edge.relation == "repeated_across_runs"]
    if not reuse_edges:
        lines.append("- None")
    for edge in reuse_edges[:50]:
        thread_ids = edge.metadata.get("thread_ids", [])
        lines.append(f"- {edge.source} repeated across runs: {', '.join(thread_ids)}")

    lines.extend(["", "## High Value Topics", ""])
    topics = [node for node in graph.nodes if node.type == "topic"]
    if not topics:
        lines.append("- None")
    for node in topics[:30]:
        score = node.metadata.get("score", "n/a")
        lines.append(f"- {node.label} (score {score})")

    if graph.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in graph.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).rstrip() + "\n"


def write_memory_graph_artifacts(thread_dir: Path, graph: MemoryGraph) -> None:
    thread_dir.mkdir(parents=True, exist_ok=True)
    (thread_dir / "memory_graph.json").write_text(
        json.dumps(_dump_model(graph), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "memory_graph.md").write_text(
        render_memory_graph_markdown(graph),
        encoding="utf-8",
    )
