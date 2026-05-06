from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import ensure_thread_dir, safe_thread_id
from deep_research_agent.settings import Settings

from .artifact_contracts import build_contracts_for_template, render_contracts_markdown
from .contracts import (
    CompiledWorkflow,
    WarningSeverity,
    WorkflowDependencyGraph,
    WorkflowInput,
    WorkflowMode,
    WorkflowStageDefinition,
    WorkflowStageType,
    WorkflowWarning,
)
from .dependency_resolver import build_dependency_graph, topological_sort
from .policies import build_policy_warnings, redact_secrets
from .registry import WorkflowTemplateRegistry, infer_mode_from_question
from .report_writer import (
    render_dependency_graph_markdown,
    render_execution_plan_markdown,
    write_json_artifact,
    write_markdown_artifact,
)

AVAILABLE_OPTIONAL_STAGE_SETTINGS = {
    WorkflowStageType.source_safety: "source_safety_enabled",
    WorkflowStageType.document_intelligence: "document_intelligence_enabled",
    WorkflowStageType.source_audit: "source_audit_enabled",
    WorkflowStageType.retrieval_indexing: "retrieval_enabled",
    WorkflowStageType.context_pack_building: "retrieval_enabled",
    WorkflowStageType.agent_control_planning: "agent_control_enabled",
    WorkflowStageType.evidence_extraction: "intelligence_source_reasoning_enabled",
    WorkflowStageType.hypothesis_testing: "hypothesis_engine_enabled",
    WorkflowStageType.temporal_analysis: "temporal_intelligence_enabled",
    WorkflowStageType.quantitative_analysis: "quantitative_intelligence_enabled",
    WorkflowStageType.synthesis: "synthesis_enabled",
    WorkflowStageType.verification: "verification_enabled",
    WorkflowStageType.evaluation: "evaluation_enabled",
    WorkflowStageType.provenance: "provenance_enabled",
    WorkflowStageType.quality_gate: "workflows_quality_gate_enabled",
}


class WorkflowCompiler:
    def __init__(self, settings: Settings, registry: WorkflowTemplateRegistry | None = None):
        self.settings = settings
        self.registry = registry or WorkflowTemplateRegistry()

    def compile(
        self,
        workflow_input: WorkflowInput,
        *,
        write_artifacts: bool = False,
    ) -> CompiledWorkflow:
        template = self.registry.select_template(workflow_input)
        thread_id = _resolve_thread_id(workflow_input)
        settings_snapshot = self._settings_snapshot(
            template.default_settings, workflow_input.settings_overrides
        )
        stages, skipped = self._resolve_stages(template.stages, workflow_input, settings_snapshot)
        graph = build_dependency_graph(stages)
        execution_order = topological_sort(stages)
        warnings = self._validation_warnings(workflow_input, stages, graph)
        workflow = CompiledWorkflow(
            workflow_id=f"wf-{uuid.uuid4().hex[:12]}",
            thread_id=thread_id,
            template_id=template.template_id,
            mode=template.mode,
            question=workflow_input.question,
            urls=[url.strip() for url in workflow_input.urls if url and url.strip()],
            stages=stages,
            dependency_graph=graph,
            artifact_contracts=build_contracts_for_template(template),
            policies=_resolve_policies(template.policies, workflow_input, settings_snapshot),
            settings_snapshot=settings_snapshot,
            execution_order=execution_order,
            skipped_stages=skipped,
            warnings=warnings,
        )
        workflow.warnings.extend(build_policy_warnings(workflow))
        if write_artifacts:
            run_dir = ensure_thread_dir(self.settings.runs_dir, workflow.thread_id)
            write_compiled_artifacts(run_dir, workflow)
        return workflow

    def infer_mode(self, workflow_input: WorkflowInput) -> WorkflowMode:
        if workflow_input.mode:
            return workflow_input.mode
        return infer_mode_from_question(workflow_input)

    def _settings_snapshot(
        self,
        template_defaults: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        base = dataclasses.asdict(self.settings)
        merged = {**base, **template_defaults, **overrides}
        return redact_secrets(
            {key: str(value) if isinstance(value, Path) else value for key, value in merged.items()}
        )

    def _resolve_stages(
        self,
        template_stages: list[WorkflowStageDefinition],
        workflow_input: WorkflowInput,
        settings_snapshot: dict[str, Any],
    ) -> tuple[list[WorkflowStageDefinition], dict[str, str]]:
        requested_gate = bool(workflow_input.run_quality_gate or workflow_input.quality_gate_id)
        stages: list[WorkflowStageDefinition] = []
        skipped: dict[str, str] = {}
        for stage in template_stages:
            if not stage.enabled_by_default:
                skipped[stage.stage_id] = "disabled_by_default"
                continue
            if stage.stage_type == WorkflowStageType.quality_gate and not requested_gate:
                policy_wants_gate = any(
                    policy.run_quality_gate
                    for policy in self.registry.select_template(workflow_input).policies
                )
                if not policy_wants_gate:
                    skipped[stage.stage_id] = "quality_gate_not_requested"
                    continue
            setting_name = AVAILABLE_OPTIONAL_STAGE_SETTINGS.get(stage.stage_type)
            if (
                setting_name
                and not stage.required
                and not bool(settings_snapshot.get(setting_name, True))
            ):
                skipped[stage.stage_id] = f"optional_subsystem_disabled:{setting_name}"
                continue
            stages.append(stage)
        present = {stage.stage_id for stage in stages}
        for stage in stages:
            stage.depends_on = [dep for dep in stage.depends_on if dep in present]
            stage.optional_depends_on = [dep for dep in stage.optional_depends_on if dep in present]
        max_stages = int(settings_snapshot.get("workflows_max_stages") or 32)
        if len(stages) > max_stages:
            for stage in stages[max_stages:]:
                skipped[stage.stage_id] = "max_stage_count_exceeded"
            stages = stages[:max_stages]
        return stages, skipped

    def _validation_warnings(
        self,
        workflow_input: WorkflowInput,
        stages: list[WorkflowStageDefinition],
        graph: WorkflowDependencyGraph,
    ) -> list[WorkflowWarning]:
        warnings: list[WorkflowWarning] = []
        mode = workflow_input.mode or infer_mode_from_question(workflow_input)
        existing_required = mode in {
            WorkflowMode.verification_only,
            WorkflowMode.evaluation_only,
            WorkflowMode.rebuild_from_artifacts,
        }
        if existing_required and not (
            workflow_input.existing_run_id
            or workflow_input.existing_thread_id
            or workflow_input.thread_id
        ):
            warnings.append(
                WorkflowWarning(
                    code="existing_run_required",
                    severity=WarningSeverity.critical,
                    message=f"{mode.value} requires an existing run/thread id.",
                )
            )
        if graph.cycles_detected:
            warnings.append(
                WorkflowWarning(
                    code="dependency_cycle",
                    severity=WarningSeverity.critical,
                    message="Workflow dependency graph contains a cycle.",
                )
            )
        if graph.missing_dependencies:
            warnings.append(
                WorkflowWarning(
                    code="missing_dependency",
                    severity=WarningSeverity.critical,
                    message="Workflow dependency graph references missing stages.",
                )
            )
        if not stages:
            warnings.append(
                WorkflowWarning(
                    code="no_executable_stages",
                    severity=WarningSeverity.critical,
                    message="Workflow has no executable stages.",
                )
            )
        return warnings


def write_compiled_artifacts(run_dir, workflow: CompiledWorkflow) -> None:
    write_json_artifact(run_dir, "workflow_compiled.json", workflow)
    write_markdown_artifact(run_dir, "workflow_compiled.md", workflow.to_markdown())
    write_json_artifact(
        run_dir,
        "workflow_execution_plan.json",
        {"execution_order": workflow.execution_order, "stages": workflow.stages},
    )
    write_markdown_artifact(
        run_dir, "workflow_execution_plan.md", render_execution_plan_markdown(workflow)
    )
    write_json_artifact(run_dir, "workflow_dependency_graph.json", workflow.dependency_graph)
    write_markdown_artifact(
        run_dir, "workflow_dependency_graph.md", render_dependency_graph_markdown(workflow)
    )
    write_json_artifact(run_dir, "workflow_warnings.json", workflow.warnings)
    write_markdown_artifact(
        run_dir,
        "workflow_warnings.md",
        "\n".join(
            f"- `{warning.severity.value}` `{warning.code}`: {warning.message}"
            for warning in workflow.warnings
        )
        or "No workflow warnings.",
    )
    write_json_artifact(run_dir, "workflow_artifact_contracts.json", workflow.artifact_contracts)
    write_markdown_artifact(
        run_dir,
        "workflow_artifact_contracts.md",
        render_contracts_markdown(workflow.artifact_contracts),
    )
    template_payload = {
        "template_id": workflow.template_id,
        "mode": workflow.mode.value,
        "stages": [stage.stage_id for stage in workflow.stages],
    }
    write_json_artifact(run_dir, "workflow_template.json", template_payload)
    write_markdown_artifact(
        run_dir,
        "workflow_template.md",
        (
            "# Workflow Template\n\n"
            f"- Template: `{workflow.template_id}`\n"
            f"- Mode: `{workflow.mode.value}`\n"
        ),
    )


def _resolve_thread_id(workflow_input: WorkflowInput) -> str:
    thread_id = (
        workflow_input.thread_id
        or workflow_input.existing_thread_id
        or workflow_input.existing_run_id
        or f"workflow-{uuid.uuid4().hex[:12]}"
    )
    return safe_thread_id(thread_id)


def _resolve_policies(policies, workflow_input: WorkflowInput, settings_snapshot: dict[str, Any]):
    out = []
    for policy in policies:
        update: dict[str, Any] = {}
        if workflow_input.run_quality_gate:
            update["run_quality_gate"] = True
            update["quality_gate_id"] = (
                workflow_input.quality_gate_id
                or policy.quality_gate_id
                or settings_snapshot.get("workflows_default_quality_gate")
                or "smoke"
            )
        elif workflow_input.quality_gate_id:
            update["run_quality_gate"] = True
            update["quality_gate_id"] = workflow_input.quality_gate_id
        copier = getattr(policy, "model_copy", None)
        out.append(copier(update=update) if callable(copier) else policy.copy(update=update))
    return out


def compile_workflow(
    workflow_input: WorkflowInput,
    *,
    settings: Settings,
    write_artifacts: bool = False,
) -> CompiledWorkflow:
    return WorkflowCompiler(settings).compile(workflow_input, write_artifacts=write_artifacts)
