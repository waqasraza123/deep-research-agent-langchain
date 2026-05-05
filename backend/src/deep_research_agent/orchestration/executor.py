from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_writer import OrchestrationArtifactWriter
from .contracts import (
    OrchestrationSummary,
    ResearchTaskGraph,
    ResearchTaskStatus,
    ResearchTaskType,
    StageError,
    StageInput,
    StageOutput,
    model_to_json_dict,
)
from .errors import TaskGraphError
from .specialists import SpecialistStage, default_specialists
from .stage_outputs import summarize_orchestration
from .task_graph import build_research_task_graph


class OrchestrationExecutor:
    def __init__(
        self,
        runs_dir: Path,
        *,
        specialists: dict[ResearchTaskType, SpecialistStage] | None = None,
        artifact_writer: OrchestrationArtifactWriter | None = None,
    ):
        self.runs_dir = runs_dir
        self.specialists = specialists or default_specialists()
        self.artifact_writer = artifact_writer or OrchestrationArtifactWriter(runs_dir)

    def build_graph(
        self,
        *,
        thread_id: str,
        question: str,
        urls: list[str] | None = None,
        strategy: Any = None,
        follow_links: bool = False,
        max_links_per_source: int = 0,
        available_source_count: int = 0,
    ) -> ResearchTaskGraph:
        return build_research_task_graph(
            thread_id=thread_id,
            question=question,
            urls=urls or [],
            strategy=strategy,
            follow_links=follow_links,
            max_links_per_source=max_links_per_source,
            available_source_count=available_source_count,
        )

    def execute(
        self,
        graph: ResearchTaskGraph,
        *,
        strategy: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> tuple[ResearchTaskGraph, list[StageOutput], OrchestrationSummary]:
        outputs: list[StageOutput] = []
        output_map: dict[str, dict[str, Any]] = {}
        by_id = graph.node_by_id()

        try:
            ordered = graph.execution_order()
        except ValueError as e:
            raise TaskGraphError(str(e)) from e

        for node in ordered:
            decision = node.decision
            failed_deps = [
                dep
                for dep in node.depends_on
                if by_id[dep].status == ResearchTaskStatus.FAILED
            ]
            if failed_deps:
                reason = "Skipped because dependency failed: " + ", ".join(failed_deps)
                output = self._skip_output(node.id, node.task_type, node.specialist_role, reason)
                node.status = ResearchTaskStatus.SKIPPED
                node.skipped_reason = reason
                node.output = output
                outputs.append(output)
                output_map[node.id] = model_to_json_dict(output)
                continue

            if decision is not None and not decision.should_run:
                output = self._skip_output(
                    node.id,
                    node.task_type,
                    node.specialist_role,
                    decision.reason,
                )
                node.status = ResearchTaskStatus.SKIPPED
                node.skipped_reason = decision.reason
                node.output = output
                outputs.append(output)
                output_map[node.id] = model_to_json_dict(output)
                continue

            specialist = self.specialists.get(node.task_type)
            if specialist is None:
                error = StageError(
                    error_type="MissingSpecialist",
                    message=f"No specialist registered for {node.task_type.value}",
                    recoverable=True,
                )
                output = self._failed_output(node.id, node.task_type, node.specialist_role, error)
                node.status = ResearchTaskStatus.FAILED
                node.error = error
                node.output = output
                outputs.append(output)
                output_map[node.id] = model_to_json_dict(output)
                continue

            node.status = ResearchTaskStatus.RUNNING
            stage_input = StageInput(
                thread_id=graph.thread_id,
                question=graph.question,
                urls=graph.urls,
                task_type=node.task_type,
                specialist_role=node.specialist_role,
                strategy=strategy,
                available_source_count=int(graph.metadata.get("available_source_count") or 0),
                follow_links=bool(graph.metadata.get("follow_links") or False),
                max_links_per_source=int(graph.metadata.get("max_links_per_source") or 0),
                previous_outputs=output_map,
                metadata={
                    **graph.metadata,
                    "routing_signals": graph.routing_signals,
                    "confidence_policy": graph.confidence_policy,
                },
            )
            try:
                output = specialist.run(node.id, stage_input)
            except Exception as e:  # noqa: BLE001 - specialists are isolated contracts.
                error = StageError(
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=True,
                )
                output = self._failed_output(node.id, node.task_type, node.specialist_role, error)
                node.status = ResearchTaskStatus.FAILED
                node.error = error
                node.output = output
                outputs.append(output)
                output_map[node.id] = model_to_json_dict(output)
                continue

            node.status = output.status
            node.output = output
            if output.error is not None:
                node.error = output.error
                node.status = ResearchTaskStatus.FAILED
            outputs.append(output)
            output_map[node.id] = model_to_json_dict(output)

        summary = summarize_orchestration(graph, outputs, [])
        if persist:
            artifacts = self.artifact_writer.write(graph=graph, outputs=outputs, summary=summary)
            summary.artifact_paths = [artifact.path for artifact in artifacts]
            self.artifact_writer.write(graph=graph, outputs=outputs, summary=summary)
        return graph, outputs, summary

    @staticmethod
    def _skip_output(
        node_id: str,
        task_type: ResearchTaskType,
        role,
        reason: str,
    ) -> StageOutput:
        return StageOutput(
            node_id=node_id,
            task_type=task_type,
            specialist_role=role,
            status=ResearchTaskStatus.SKIPPED,
            stage_summary=f"Skipped: {reason}",
            findings=[],
            warnings=[],
            required_next_steps=[],
            confidence_score=0.0,
            artifact_updates={},
            model_instruction_block="",
            skipped_reason=reason,
        )

    @staticmethod
    def _failed_output(
        node_id: str,
        task_type: ResearchTaskType,
        role,
        error: StageError,
    ) -> StageOutput:
        return StageOutput(
            node_id=node_id,
            task_type=task_type,
            specialist_role=role,
            status=ResearchTaskStatus.FAILED,
            stage_summary=f"Failed: {error.error_type}: {error.message}",
            findings=[],
            warnings=[f"Stage failed but orchestration continued: {error.message}"],
            required_next_steps=[f"Review failed stage {node_id} before relying on its coverage."],
            confidence_score=0.0,
            artifact_updates={},
            model_instruction_block="",
            error=error,
        )
