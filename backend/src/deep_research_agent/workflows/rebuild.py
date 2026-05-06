from __future__ import annotations

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.settings import Settings

from .compiler import WorkflowCompiler
from .contracts import WorkflowInput, WorkflowMode, WorkflowRebuildRequest, WorkflowRebuildResult
from .execution_context import WorkflowExecutionContext
from .stage_executor import execute_workflow


def rebuild_workflow(
    request: WorkflowRebuildRequest,
    *,
    settings: Settings,
    service=None,
) -> WorkflowRebuildResult:
    run_dir = ensure_thread_dir(settings.runs_dir, request.thread_id)
    if not run_dir.exists():
        return WorkflowRebuildResult(
            thread_id=request.thread_id,
            mode=request.mode,
            stages_requested=request.stages,
            stages_failed=request.stages or ["existing_run"],
            warnings=["Existing run directory does not exist."],
        )
    if not request.allow_source_refetch and request.mode in {
        WorkflowMode.rebuild_from_artifacts,
        WorkflowMode.verification_only,
        WorkflowMode.evaluation_only,
    }:
        request.settings_overrides["workflows_allow_external_network"] = False
    workflow_input = WorkflowInput(
        question="Rebuild workflow artifacts from existing run.",
        thread_id=request.thread_id,
        existing_run_id=request.thread_id,
        mode=request.mode,
        settings_overrides=request.settings_overrides,
        run_now=True,
    )
    compiled = WorkflowCompiler(settings).compile(workflow_input, write_artifacts=True)
    if request.stages:
        selected = set(request.stages)
        compiled.stages = [
            stage
            for stage in compiled.stages
            if stage.stage_id in selected or stage.stage_type.value == "finalization"
        ]
        compiled.execution_order = [
            stage_id
            for stage_id in compiled.execution_order
            if stage_id in selected or stage_id == "finalization"
        ]
    context = WorkflowExecutionContext(
        compiled, settings=settings, service=service, run_dir=run_dir
    )
    result = execute_workflow(compiled, context)
    return WorkflowRebuildResult(
        thread_id=request.thread_id,
        mode=request.mode,
        stages_requested=request.stages,
        stages_rebuilt=[
            stage.stage_id for stage in result.stages if stage.status.value == "completed"
        ],
        stages_skipped=[
            stage.stage_id for stage in result.stages if stage.status.value == "skipped"
        ],
        stages_failed=[stage.stage_id for stage in result.stages if stage.status.value == "failed"],
        generated_artifacts=result.generated_artifacts,
        warnings=[warning.message for warning in result.warnings],
    )
