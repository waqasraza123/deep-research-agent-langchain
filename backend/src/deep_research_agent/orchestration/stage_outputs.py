from __future__ import annotations

import json
from typing import Any

from .contracts import (
    OrchestrationSummary,
    ResearchTaskGraph,
    ResearchTaskStatus,
    StageError,
    StageOutput,
    model_to_json_dict,
)


def stage_outputs_payload(outputs: list[StageOutput]) -> list[dict[str, Any]]:
    return [model_to_json_dict(output) for output in outputs]


def stage_outputs_json(outputs: list[StageOutput]) -> str:
    return json.dumps(stage_outputs_payload(outputs), indent=2, ensure_ascii=False) + "\n"


def agent_instruction_block(graph: ResearchTaskGraph, outputs: list[StageOutput]) -> str:
    blocks = [
        output.model_instruction_block.strip()
        for output in outputs
        if output.model_instruction_block.strip()
    ]
    if not blocks:
        return ""
    header = [
        "Adaptive orchestration instructions:",
        f"- Graph ID: {graph.graph_id}",
        f"- Confidence policy: {graph.confidence_policy}",
        "- Treat these as planning constraints, not factual conclusions.",
        "",
    ]
    return "\n".join(header + blocks).strip()


def summarize_orchestration(
    graph: ResearchTaskGraph,
    outputs: list[StageOutput],
    artifact_paths: list[str],
) -> OrchestrationSummary:
    executed = [
        output.node_id for output in outputs if output.status == ResearchTaskStatus.SUCCEEDED
    ]
    skipped = {
        node.id: node.skipped_reason or (node.decision.reason if node.decision else "Skipped")
        for node in graph.nodes
        if node.status.value == "skipped"
    }
    failed: dict[str, StageError] = {
        node.id: node.error for node in graph.nodes if node.error is not None
    }
    warnings: list[str] = []
    next_steps: list[str] = []
    for output in outputs:
        warnings.extend(output.warnings)
        next_steps.extend(output.required_next_steps)
    status = ResearchTaskStatus.FAILED if failed else ResearchTaskStatus.SUCCEEDED
    return OrchestrationSummary(
        thread_id=graph.thread_id,
        graph_id=graph.graph_id,
        status=status,
        executed_nodes=executed,
        skipped_nodes=skipped,
        failed_nodes=failed,
        warnings=list(dict.fromkeys(warnings)),
        required_next_steps=list(dict.fromkeys(next_steps)),
        agent_instruction_block=agent_instruction_block(graph, outputs),
        confidence_policy=graph.confidence_policy,
        artifact_paths=artifact_paths,
    )
