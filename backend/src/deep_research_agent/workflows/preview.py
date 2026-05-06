from __future__ import annotations

from deep_research_agent.settings import Settings

from .compiler import WorkflowCompiler
from .contracts import WorkflowInput, WorkflowPreview


def preview_workflow(
    workflow_input: WorkflowInput,
    *,
    settings: Settings,
    write_artifacts: bool = False,
) -> WorkflowPreview:
    compiler = WorkflowCompiler(settings)
    compiled = compiler.compile(workflow_input, write_artifacts=write_artifacts)
    complexity = (
        "high" if len(compiled.stages) >= 10 else "medium" if len(compiled.stages) >= 6 else "low"
    )
    return WorkflowPreview(
        mode=compiled.mode,
        template_id=compiled.template_id,
        question=compiled.question,
        selected_template={"template_id": compiled.template_id, "mode": compiled.mode.value},
        stages=compiled.stages,
        execution_order=compiled.execution_order,
        expected_artifacts=[contract.artifact_name for contract in compiled.artifact_contracts],
        policies=compiled.policies,
        estimated_complexity=complexity,
        skipped_stages=compiled.skipped_stages,
        warnings=compiled.warnings,
    )
