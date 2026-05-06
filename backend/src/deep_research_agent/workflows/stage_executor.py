from __future__ import annotations

import json
from typing import Any

from deep_research_agent.artifacts import ensure_required_artifacts
from deep_research_agent.evaluation import rebuild_evaluation_artifacts
from deep_research_agent.evaluation_lab.contracts import QualityGateRunRequest
from deep_research_agent.evaluation_lab.gate_runner import QualityGateRunner
from deep_research_agent.evidence import rebuild_evidence_artifacts
from deep_research_agent.provenance import refresh_provenance_artifacts
from deep_research_agent.runtime.mock_model import write_mock_research_artifacts
from deep_research_agent.source_audit import (
    audit_sources_from_manifest,
    write_source_audit_artifacts,
)
from deep_research_agent.source_safety import (
    assess_sources_from_manifest,
    write_source_safety_artifacts,
)
from deep_research_agent.synthesis import rebuild_synthesis_artifacts
from deep_research_agent.temporal import rebuild_temporal_artifacts
from deep_research_agent.verification import rebuild_verification_artifacts

from .artifact_contracts import (
    validate_artifact_contracts,
    write_artifact_contract_outputs,
)
from .contracts import (
    CompiledWorkflow,
    WarningSeverity,
    WorkflowExecutionResult,
    WorkflowExecutionStatus,
    WorkflowFailureBehavior,
    WorkflowStageDefinition,
    WorkflowStageExecution,
    WorkflowStageStatus,
    WorkflowStageType,
    model_to_plain,
    now_iso_utc,
)
from .execution_context import WorkflowExecutionContext
from .manifest import write_manifest_artifacts
from .policies import (
    evaluate_post_execution_policies,
    evaluate_stage_policies,
    write_policy_artifacts,
)
from .report_writer import (
    render_stage_results_markdown,
    write_json_artifact,
    write_markdown_artifact,
)


def execute_workflow(
    compiled_workflow: CompiledWorkflow,
    context: WorkflowExecutionContext,
) -> WorkflowExecutionResult:
    started = now_iso_utc()
    warnings = list(compiled_workflow.warnings)
    stages: list[WorkflowStageExecution] = []
    quality_gate_result: dict[str, Any] | None = None
    status = WorkflowExecutionStatus.running
    write_policy_artifacts(context.run_dir, compiled_workflow, warnings)
    for stage_id in compiled_workflow.execution_order:
        stage = next(stage for stage in compiled_workflow.stages if stage.stage_id == stage_id)
        stage_warnings = evaluate_stage_policies(stage, compiled_workflow)
        warnings.extend(stage_warnings)
        if any(
            w.severity == WarningSeverity.critical and w.code in {"stage_denied"}
            for w in stage_warnings
        ):
            execution = fail_stage(stage, context, "Stage denied by workflow policy.")
        elif context.dry_run:
            execution = skip_stage(stage, context, "dry_run")
        elif _upstream_failed(stage, stages):
            execution = skip_stage(stage, context, "upstream_required_stage_failed")
        elif stage.skip_if_artifacts_exist and all(
            (context.run_dir / artifact).exists() for artifact in stage.skip_if_artifacts_exist
        ):
            execution = skip_stage(stage, context, "skip_if_artifacts_exist")
        else:
            execution = execute_stage(stage, context)
        stages.append(execution)
        context.stage_outputs[stage.stage_id] = execution
        context.refresh_artifacts()
        if execution.status == WorkflowStageStatus.failed:
            if stage.failure_behavior == WorkflowFailureBehavior.fail_workflow or (
                stage.required and context.settings.workflows_fail_on_required_stage_failure
            ):
                status = WorkflowExecutionStatus.failed
                break
            context.degraded = True
        if stage.stage_type == WorkflowStageType.quality_gate and execution.output_artifacts:
            path = context.run_dir / "workflow_quality_gate_result.json"
            if path.exists():
                try:
                    quality_gate_result = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    quality_gate_result = {"status": "error"}
    write_json_artifact(context.run_dir, "workflow_stage_results.json", stages)
    write_markdown_artifact(
        context.run_dir, "workflow_stage_results.md", render_stage_results_markdown(stages)
    )
    validation = validate_artifact_contracts(
        context.run_dir,
        compiled_workflow.artifact_contracts,
        stage_statuses={stage.stage_id: stage.status for stage in stages},
    )
    missing = validation.missing_required
    write_artifact_contract_outputs(
        context.run_dir, compiled_workflow.artifact_contracts, validation
    )
    if status != WorkflowExecutionStatus.failed:
        if validation.status.value == "failed":
            status = WorkflowExecutionStatus.failed
        elif context.degraded:
            status = WorkflowExecutionStatus.degraded
        elif warnings or validation.status.value == "warning":
            status = WorkflowExecutionStatus.completed_with_warnings
        else:
            status = WorkflowExecutionStatus.completed
    errors = [error for stage in stages for error in stage.errors]
    result = WorkflowExecutionResult(
        workflow_id=compiled_workflow.workflow_id,
        thread_id=compiled_workflow.thread_id,
        mode=compiled_workflow.mode,
        status=status,
        started_at=started,
        completed_at=now_iso_utc(),
        stages=stages,
        generated_artifacts=sorted(context.available_artifacts),
        missing_artifacts=missing,
        warnings=warnings,
        errors=errors,
        degraded=context.degraded or status == WorkflowExecutionStatus.degraded,
        quality_gate_result=quality_gate_result,
        summary=f"Workflow {compiled_workflow.mode.value} finished with status {status.value}.",
    )
    result.readiness = evaluate_post_execution_policies(result)
    write_manifest_artifacts(context.run_dir, compiled_workflow, result)
    return result


def execute_stage(
    stage: WorkflowStageDefinition,
    context: WorkflowExecutionContext,
) -> WorkflowStageExecution:
    execution = WorkflowStageExecution(
        workflow_id=context.compiled_workflow.workflow_id,
        thread_id=context.compiled_workflow.thread_id,
        stage_id=stage.stage_id,
        stage_type=stage.stage_type,
        status=WorkflowStageStatus.running,
        started_at=now_iso_utc(),
        attempts=1,
        input_artifacts=stage.input_artifacts,
    )
    try:
        outputs = _execute_stage_impl(stage, context)
        execution.status = WorkflowStageStatus.completed
        execution.completed_at = now_iso_utc()
        execution.output_artifacts = outputs
        execution.metrics = {"artifact_count": len(outputs)}
        return execution
    except SkippedStage as e:
        execution.status = WorkflowStageStatus.skipped
        execution.skipped_at = now_iso_utc()
        execution.warnings.append(str(e))
        return execution
    except DegradedStage as e:
        context.degraded = True
        execution.status = WorkflowStageStatus.degraded
        execution.completed_at = now_iso_utc()
        execution.warnings.append(str(e))
        return execution
    except Exception as e:
        execution.status = WorkflowStageStatus.failed
        execution.failed_at = now_iso_utc()
        execution.errors.append(f"{type(e).__name__}: {e}")
        return execution


def skip_stage(
    stage: WorkflowStageDefinition, context: WorkflowExecutionContext, reason: str
) -> WorkflowStageExecution:
    write_json_artifact(
        context.run_dir,
        f"workflow_skipped_{stage.stage_id}.json",
        {"stage_id": stage.stage_id, "reason": reason, "skipped_at": now_iso_utc()},
    )
    return WorkflowStageExecution(
        workflow_id=context.compiled_workflow.workflow_id,
        thread_id=context.compiled_workflow.thread_id,
        stage_id=stage.stage_id,
        stage_type=stage.stage_type,
        status=WorkflowStageStatus.skipped,
        skipped_at=now_iso_utc(),
        warnings=[reason],
        output_artifacts=[f"workflow_skipped_{stage.stage_id}.json"],
    )


def fail_stage(
    stage: WorkflowStageDefinition, context: WorkflowExecutionContext, error: str
) -> WorkflowStageExecution:
    return WorkflowStageExecution(
        workflow_id=context.compiled_workflow.workflow_id,
        thread_id=context.compiled_workflow.thread_id,
        stage_id=stage.stage_id,
        stage_type=stage.stage_type,
        status=WorkflowStageStatus.failed,
        failed_at=now_iso_utc(),
        errors=[error],
    )


def complete_stage(
    stage: WorkflowStageDefinition, context: WorkflowExecutionContext, outputs: list[str]
) -> WorkflowStageExecution:
    return WorkflowStageExecution(
        workflow_id=context.compiled_workflow.workflow_id,
        thread_id=context.compiled_workflow.thread_id,
        stage_id=stage.stage_id,
        stage_type=stage.stage_type,
        status=WorkflowStageStatus.completed,
        completed_at=now_iso_utc(),
        output_artifacts=outputs,
    )


def execute_until_stage(
    compiled_workflow: CompiledWorkflow,
    context: WorkflowExecutionContext,
    stage_id: str,
) -> WorkflowExecutionResult:
    limited = compiled_workflow.model_copy(
        update={
            "execution_order": compiled_workflow.execution_order[
                : compiled_workflow.execution_order.index(stage_id) + 1
            ]
        }
    )
    return execute_workflow(limited, context)


def rebuild_stage_from_artifacts(
    stage: WorkflowStageDefinition, context: WorkflowExecutionContext
) -> WorkflowStageExecution:
    return execute_stage(stage, context)


def _execute_stage_impl(
    stage: WorkflowStageDefinition, context: WorkflowExecutionContext
) -> list[str]:
    stage_type = stage.stage_type
    if stage_type == WorkflowStageType.input_snapshot:
        return _input_snapshot(context)
    if stage_type == WorkflowStageType.request_analysis:
        return _request_analysis(context)
    if stage_type == WorkflowStageType.protocol_selection:
        return _simple_json_md(
            context, "workflow_protocol_selection", {"policy": "template_default"}
        )
    if stage_type == WorkflowStageType.source_discovery:
        raise SkippedStage("source discovery adapter unavailable or disabled")
    if stage_type == WorkflowStageType.source_fetching:
        return _source_fetching(context)
    if stage_type == WorkflowStageType.source_safety:
        return _source_safety(context, required=stage.required)
    if stage_type == WorkflowStageType.document_intelligence:
        raise SkippedStage(
            "document intelligence rebuild adapter is not available in workflow runtime"
        )
    if stage_type == WorkflowStageType.source_audit:
        return _source_audit(context, required=stage.required)
    if stage_type in {
        WorkflowStageType.retrieval_indexing,
        WorkflowStageType.context_pack_building,
    }:
        raise SkippedStage("retrieval/context-pack adapter unavailable in workflow runtime")
    if stage_type == WorkflowStageType.agent_control_planning:
        raise SkippedStage("agent control planning adapter unavailable in workflow runtime")
    if stage_type == WorkflowStageType.agent_execution:
        return _agent_execution(context, required=stage.required)
    if stage_type == WorkflowStageType.artifact_backfill:
        return _artifact_backfill(context)
    if stage_type == WorkflowStageType.evidence_extraction:
        return _rebuild_artifact_set(
            context, rebuild_evidence_artifacts, ["evidence_ledger.json", "evidence_ledger.md"]
        )
    if stage_type == WorkflowStageType.hypothesis_testing:
        raise SkippedStage("hypothesis adapter unavailable in workflow runtime")
    if stage_type == WorkflowStageType.temporal_analysis:
        return _temporal(context)
    if stage_type == WorkflowStageType.quantitative_analysis:
        raise SkippedStage("quantitative adapter unavailable in workflow runtime")
    if stage_type == WorkflowStageType.synthesis:
        return _synthesis(context)
    if stage_type == WorkflowStageType.verification:
        return _rebuild_artifact_set(
            context,
            rebuild_verification_artifacts,
            ["verification_results.json", "verification_summary.md"],
        )
    if stage_type == WorkflowStageType.evaluation:
        return _rebuild_artifact_set(
            context, rebuild_evaluation_artifacts, ["evaluation.json", "evaluation.md"]
        )
    if stage_type == WorkflowStageType.quality_gate:
        return _quality_gate(context)
    if stage_type == WorkflowStageType.provenance:
        refresh_provenance_artifacts(context.settings.runs_dir, context.compiled_workflow.thread_id)
        return ["artifact_manifest.json", "dependency_graph.json", "reproducibility.json"]
    if stage_type == WorkflowStageType.finalization:
        return _finalization(context)
    raise SkippedStage(f"No adapter for stage type {stage_type.value}")


def _input_snapshot(context: WorkflowExecutionContext) -> list[str]:
    payload = {
        "workflow_id": context.compiled_workflow.workflow_id,
        "thread_id": context.compiled_workflow.thread_id,
        "mode": context.compiled_workflow.mode.value,
        "question": context.compiled_workflow.question,
        "urls": context.compiled_workflow.urls,
        "generated_at": now_iso_utc(),
    }
    write_json_artifact(context.run_dir, "workflow_input_snapshot.json", payload)
    return ["workflow_input_snapshot.json"]


def _request_analysis(context: WorkflowExecutionContext) -> list[str]:
    question = context.compiled_workflow.question
    words = question.split()
    payload = {
        "question_length": len(question),
        "word_count": len(words),
        "url_count": len(context.compiled_workflow.urls),
        "time_sensitive": any(
            term in question.lower() for term in ("current", "latest", "today", "version")
        ),
        "comparison": any(term in question.lower() for term in ("compare", " vs ", "versus")),
    }
    return _simple_json_md(context, "workflow_request_analysis", payload)


def _source_fetching(context: WorkflowExecutionContext) -> list[str]:
    policy_network_allowed = all(
        policy.external_network_allowed for policy in context.compiled_workflow.policies
    )
    if not policy_network_allowed:
        if not (context.run_dir / "sources.json").exists():
            write_json_artifact(
                context.run_dir,
                "sources.json",
                [
                    {
                        "source_id": "S1",
                        "url": "workflow://no-refetch",
                        "ok": False,
                        "summary": "Source refetch denied by workflow policy.",
                    }
                ],
            )
        return _simple_json_md(
            context,
            "workflow_source_fetching_summary",
            {"status": "skipped", "reason": "external_network_denied"},
        )
    if not (context.run_dir / "sources.json").exists():
        sources = [
            {
                "source_id": f"S{idx}",
                "url": url,
                "ok": False,
                "summary": "URL queued for agent/tool fetching.",
            }
            for idx, url in enumerate(context.compiled_workflow.urls, start=1)
        ]
        if not sources:
            sources = [
                {
                    "source_id": "S1",
                    "url": "workflow://no-source-provided",
                    "ok": False,
                    "summary": "No source URL supplied.",
                }
            ]
        write_json_artifact(context.run_dir, "sources.json", sources)
    return _simple_json_md(
        context,
        "workflow_source_fetching_summary",
        {"status": "prepared", "url_count": len(context.compiled_workflow.urls)},
    )


def _source_safety(context: WorkflowExecutionContext, *, required: bool) -> list[str]:
    if not context.has_source_safety():
        if required:
            raise RuntimeError("source safety subsystem unavailable")
        raise SkippedStage("source safety subsystem disabled")
    if not (context.run_dir / "sources.json").exists():
        raise SkippedStage("sources.json unavailable")
    batch = assess_sources_from_manifest(
        thread_dir=context.run_dir,
        thread_id=context.compiled_workflow.thread_id,
        question=context.compiled_workflow.question,
    )
    return write_source_safety_artifacts(context.run_dir, batch)


def _source_audit(context: WorkflowExecutionContext, *, required: bool) -> list[str]:
    if not context.has_source_audit():
        if required:
            raise RuntimeError("source audit subsystem unavailable")
        raise SkippedStage("source audit subsystem disabled")
    batch = audit_sources_from_manifest(
        context.run_dir / "sources.json",
        question=context.compiled_workflow.question,
        thread_id=context.compiled_workflow.thread_id,
    )
    return write_source_audit_artifacts(context.run_dir, batch)


def _agent_execution(context: WorkflowExecutionContext, *, required: bool) -> list[str]:
    policies = context.compiled_workflow.policies
    mock_allowed = any(policy.mock_allowed for policy in policies) or bool(
        context.settings.workflows_allow_mock_agent
    )
    provider = context.compiled_workflow.settings_snapshot.get("model_provider")
    if (
        provider == "mock"
        or context.mock_mode
        or (mock_allowed and not _service_available(context))
    ):
        write_mock_research_artifacts(
            thread_dir=context.run_dir,
            thread_id=context.compiled_workflow.thread_id,
            question=context.compiled_workflow.question,
            sources_meta=context.read_sources(),
        )
        return ["plan.md", "notes.md", "sources.json", "report.md", "metadata.json"]
    if not _service_available(context):
        if required:
            raise RuntimeError("agent service unavailable and mock is not allowed")
        raise SkippedStage("agent service unavailable")
    prompt = context.compiled_workflow.question
    if context.compiled_workflow.urls:
        prompt += "\n\nSources:\n" + "\n".join(f"- {url}" for url in context.compiled_workflow.urls)
    agent = context.service.build_agent(context.compiled_workflow.thread_id)
    agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": context.compiled_workflow.thread_id}},
    )
    ensure_required_artifacts(context.settings.runs_dir, context.compiled_workflow.thread_id)
    return ["plan.md", "notes.md", "sources.json", "report.md"]


def _artifact_backfill(context: WorkflowExecutionContext) -> list[str]:
    warnings = ensure_required_artifacts(
        context.settings.runs_dir, context.compiled_workflow.thread_id
    )
    if warnings:
        context.warnings.extend(warnings)
    return ["plan.md", "notes.md", "sources.json", "report.md"]


def _temporal(context: WorkflowExecutionContext) -> list[str]:
    rebuild_temporal_artifacts(
        context.run_dir,
        thread_id=context.compiled_workflow.thread_id,
        question=context.compiled_workflow.question,
        include_claims=True,
    )
    return ["temporal_analysis.json", "temporal_analysis.md"]


def _synthesis(context: WorkflowExecutionContext) -> list[str]:
    rebuild_synthesis_artifacts(context.run_dir, thread_id=context.compiled_workflow.thread_id)
    return ["synthesis.json", "synthesis.md", "report.md"]


def _rebuild_artifact_set(
    context: WorkflowExecutionContext, func, expected: list[str]
) -> list[str]:
    func(context.run_dir, thread_id=context.compiled_workflow.thread_id)
    return [artifact for artifact in expected if (context.run_dir / artifact).exists()] or expected


def _quality_gate(context: WorkflowExecutionContext) -> list[str]:
    if not context.has_quality_gates():
        raise SkippedStage("evaluation lab quality gates disabled")
    policy_gate = next(
        (
            policy
            for policy in context.compiled_workflow.policies
            if policy.run_quality_gate or policy.quality_gate_id
        ),
        None,
    )
    gate_id = (
        policy_gate.quality_gate_id if policy_gate else None
    ) or context.settings.workflows_default_quality_gate
    runner = context.quality_gate_runner or QualityGateRunner(context.settings)
    result = runner.run_gate(
        QualityGateRunRequest(
            gate_id=gate_id or "smoke",
            dry_run=False,
            use_mock_agent=True,
            use_offline_fetcher=True,
            metadata={
                "workflow_id": context.compiled_workflow.workflow_id,
                "thread_id": context.compiled_workflow.thread_id,
            },
        )
    )
    payload = model_to_plain(result)
    write_json_artifact(context.run_dir, "workflow_quality_gate_result.json", payload)
    status = str(payload.get("status") or "")
    if status not in {"passed", "skipped"}:
        context.degraded = True
        if context.settings.workflows_fail_on_quality_gate_failure:
            raise RuntimeError(f"quality gate {gate_id} failed with status {status}")
    return ["workflow_quality_gate_result.json"]


def _finalization(context: WorkflowExecutionContext) -> list[str]:
    validation = validate_artifact_contracts(
        context.run_dir, context.compiled_workflow.artifact_contracts
    )
    write_artifact_contract_outputs(
        context.run_dir, context.compiled_workflow.artifact_contracts, validation
    )
    return [
        "workflow_artifact_validation.json",
        "workflow_artifact_validation.md",
        "workflow_artifact_contracts.json",
        "workflow_artifact_contracts.md",
    ]


def _simple_json_md(
    context: WorkflowExecutionContext, stem: str, payload: dict[str, Any]
) -> list[str]:
    json_name = f"{stem}.json"
    md_name = f"{stem}.md"
    write_json_artifact(context.run_dir, json_name, payload)
    lines = [f"# {stem.replace('_', ' ').title()}", ""]
    for key, value in payload.items():
        lines.append(f"- **{key}**: {value}")
    write_markdown_artifact(context.run_dir, md_name, "\n".join(lines))
    return [json_name, md_name]


def _service_available(context: WorkflowExecutionContext) -> bool:
    return context.service is not None and hasattr(context.service, "build_agent")


def _upstream_failed(stage: WorkflowStageDefinition, stages: list[WorkflowStageExecution]) -> bool:
    failed = {item.stage_id for item in stages if item.status == WorkflowStageStatus.failed}
    return any(dep in failed for dep in stage.depends_on)


class SkippedStage(Exception):
    pass


class DegradedStage(Exception):
    pass
