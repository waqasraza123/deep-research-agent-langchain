from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import Artifact, artifact_abs_path, ensure_thread_dir

from .contracts import OrchestrationSummary, ResearchTaskGraph, StageOutput
from .stage_outputs import stage_outputs_json

ORCHESTRATION_ARTIFACTS = (
    "task_graph.json",
    "task_graph.md",
    "stage_outputs.json",
    "specialist_findings.md",
    "orchestration_summary.json",
    "orchestration_summary.md",
)


def _artifact(runs_dir: Path, thread_id: str, rel_path: str) -> Artifact:
    td = ensure_thread_dir(runs_dir, thread_id)
    path = td / rel_path
    return Artifact(path=rel_path, size_bytes=path.stat().st_size, mtime_epoch=path.stat().st_mtime)


def task_graph_markdown(graph: ResearchTaskGraph) -> str:
    lines = [
        "# Research Task Graph",
        "",
        f"- Graph ID: `{graph.graph_id}`",
        f"- Thread ID: `{graph.thread_id}`",
        f"- Confidence policy: `{graph.confidence_policy}`",
        f"- Routing signals: {', '.join(graph.routing_signals) or 'none'}",
        "",
        "## Nodes",
        "",
    ]
    for node in graph.nodes:
        decision = node.decision
        should_run = "run" if decision and decision.should_run else "skip"
        lines.extend(
            [
                f"### {node.id}",
                "",
                f"- Task: `{node.task_type.value}`",
                f"- Specialist: `{node.specialist_role.value}`",
                f"- Status: `{node.status.value}`",
                f"- Decision: `{should_run}`",
                f"- Reason: {decision.reason if decision else 'No decision recorded.'}",
            ]
        )
        if node.depends_on:
            lines.append(f"- Depends on: {', '.join(f'`{dep}`' for dep in node.depends_on)}")
        if node.skipped_reason:
            lines.append(f"- Skipped reason: {node.skipped_reason}")
        if node.error:
            lines.append(f"- Error: {node.error.error_type}: {node.error.message}")
        lines.append("")

    lines.extend(["## Edges", ""])
    for edge in graph.edges:
        lines.append(f"- `{edge.from_node}` -> `{edge.to_node}`: {edge.reason}")
    lines.append("")
    return "\n".join(lines)


def specialist_findings_markdown(outputs: list[StageOutput]) -> str:
    lines = ["# Specialist Findings", ""]
    for output in outputs:
        lines.extend(
            [
                f"## {output.task_type.value}",
                "",
                f"- Node: `{output.node_id}`",
                f"- Specialist: `{output.specialist_role.value}`",
                f"- Status: `{output.status.value}`",
                f"- Confidence: `{output.confidence_score:.2f}`",
                "",
                output.stage_summary,
                "",
            ]
        )
        if output.findings:
            lines.extend(["### Findings", ""])
            lines.extend(f"- {item}" for item in output.findings)
            lines.append("")
        if output.warnings:
            lines.extend(["### Warnings", ""])
            lines.extend(f"- {item}" for item in output.warnings)
            lines.append("")
        if output.required_next_steps:
            lines.extend(["### Required Next Steps", ""])
            lines.extend(f"- {item}" for item in output.required_next_steps)
            lines.append("")
    return "\n".join(lines)


def orchestration_summary_markdown(summary: OrchestrationSummary) -> str:
    lines = [
        "# Orchestration Summary",
        "",
        f"- Thread ID: `{summary.thread_id}`",
        f"- Graph ID: `{summary.graph_id}`",
        f"- Status: `{summary.status.value}`",
        f"- Confidence policy: `{summary.confidence_policy}`",
        "",
        "## Executed Nodes",
        "",
    ]
    lines.extend(f"- `{node_id}`" for node_id in summary.executed_nodes)
    if not summary.executed_nodes:
        lines.append("- None")
    if summary.skipped_nodes:
        lines.extend(["", "## Skipped Nodes", ""])
        lines.extend(
            f"- `{node_id}`: {reason}" for node_id, reason in summary.skipped_nodes.items()
        )
    if summary.failed_nodes:
        lines.extend(["", "## Failed Nodes", ""])
        for node_id, error in summary.failed_nodes.items():
            lines.append(f"- `{node_id}`: {error.error_type}: {error.message}")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    if summary.required_next_steps:
        lines.extend(["", "## Required Next Steps", ""])
        lines.extend(f"- {step}" for step in summary.required_next_steps)
    if summary.agent_instruction_block:
        lines.extend(
            [
                "",
                "## Agent Instruction Block",
                "",
                "```text",
                summary.agent_instruction_block,
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


class OrchestrationArtifactWriter:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def write(
        self,
        *,
        graph: ResearchTaskGraph,
        outputs: list[StageOutput],
        summary: OrchestrationSummary,
    ) -> list[Artifact]:
        thread_id = graph.thread_id
        ensure_thread_dir(self.runs_dir, thread_id)
        files: dict[str, str] = {
            "task_graph.json": graph.to_json(),
            "task_graph.md": task_graph_markdown(graph),
            "stage_outputs.json": stage_outputs_json(outputs),
            "specialist_findings.md": specialist_findings_markdown(outputs),
            "orchestration_summary.json": json.dumps(
                summary.to_json_dict(), indent=2, ensure_ascii=False
            )
            + "\n",
            "orchestration_summary.md": orchestration_summary_markdown(summary),
        }
        for rel_path, content in files.items():
            path = artifact_abs_path(self.runs_dir, thread_id, rel_path)
            path.write_text(content, encoding="utf-8")
        return [_artifact(self.runs_dir, thread_id, rel_path) for rel_path in sorted(files)]


def read_orchestration_json(runs_dir: Path, thread_id: str, rel_path: str) -> dict[str, Any]:
    if rel_path not in {"task_graph.json", "stage_outputs.json", "orchestration_summary.json"}:
        raise ValueError("Unsupported orchestration artifact")
    path = artifact_abs_path(runs_dir, thread_id, rel_path)
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(rel_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"items": data} if isinstance(data, list) else data
