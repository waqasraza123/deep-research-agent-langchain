from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import model_to_plain


def write_json_artifact(run_dir: Path, rel_path: str, payload: Any) -> str:
    path = _safe_child(run_dir, rel_path)
    path.write_text(
        json.dumps(model_to_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rel_path


def write_markdown_artifact(run_dir: Path, rel_path: str, markdown: str) -> str:
    path = _safe_child(run_dir, rel_path)
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return rel_path


def render_execution_plan_markdown(workflow) -> str:
    lines = ["# Workflow Execution Plan", "", f"Workflow: `{workflow.workflow_id}`", ""]
    for idx, stage_id in enumerate(workflow.execution_order, start=1):
        stage = next((item for item in workflow.stages if item.stage_id == stage_id), None)
        stage_type = stage.stage_type.value if stage else "unknown"
        required = "required" if stage and stage.required else "optional"
        lines.append(f"{idx}. `{stage_id}` ({stage_type}, {required})")
    if workflow.skipped_stages:
        lines.extend(["", "## Skipped"])
        for stage_id, reason in workflow.skipped_stages.items():
            lines.append(f"- `{stage_id}`: {reason}")
    return "\n".join(lines)


def render_dependency_graph_markdown(workflow) -> str:
    lines = ["# Workflow Dependency Graph", ""]
    for source, target in workflow.dependency_graph.edges:
        lines.append(f"- `{source}` -> `{target}`")
    if workflow.dependency_graph.cycles_detected:
        lines.extend(["", "## Cycles"])
        for cycle in workflow.dependency_graph.cycles_detected:
            lines.append("- " + " -> ".join(f"`{item}`" for item in cycle))
    if workflow.dependency_graph.missing_dependencies:
        lines.extend(["", "## Missing Dependencies"])
        for item in workflow.dependency_graph.missing_dependencies:
            lines.append(f"- `{item['stage_id']}` requires `{item['missing_dependency']}`")
    return "\n".join(lines)


def render_stage_results_markdown(stages) -> str:
    lines = ["# Workflow Stage Results", ""]
    for stage in stages:
        lines.append(f"- `{stage.stage_id}`: {stage.status.value}")
        for error in stage.errors:
            lines.append(f"  - Error: {error}")
        for warning in stage.warnings:
            lines.append(f"  - Warning: {warning}")
    return "\n".join(lines)


def _safe_child(run_dir: Path, rel_path: str) -> Path:
    if rel_path.startswith("/") or "\\" in rel_path or ".." in rel_path.split("/"):
        raise ValueError(f"Unsafe workflow artifact path: {rel_path}")
    path = (run_dir / rel_path).resolve()
    if not str(path).startswith(str(run_dir.resolve())):
        raise ValueError(f"Unsafe workflow artifact path: {rel_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
