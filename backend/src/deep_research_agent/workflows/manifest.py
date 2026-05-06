from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifact_contracts import list_generated_artifacts
from .contracts import (
    CompiledWorkflow,
    WorkflowExecutionResult,
    WorkflowManifest,
    WorkflowStageStatus,
    model_to_plain,
    now_iso_utc,
)
from .policies import settings_fingerprint
from .report_writer import write_json_artifact, write_markdown_artifact


def build_manifest(
    workflow: CompiledWorkflow,
    result: WorkflowExecutionResult,
    run_dir: Path,
) -> WorkflowManifest:
    stages = result.stages
    quality_gate_status = None
    if result.quality_gate_result:
        quality_gate_status = str(result.quality_gate_result.get("status") or "")
    return WorkflowManifest(
        workflow_id=workflow.workflow_id,
        thread_id=workflow.thread_id,
        template_id=workflow.template_id,
        mode=workflow.mode,
        input_fingerprint=_fingerprint(
            {
                "question": workflow.question,
                "urls": workflow.urls,
                "mode": workflow.mode.value,
                "template_id": workflow.template_id,
            }
        ),
        settings_fingerprint=settings_fingerprint(workflow.settings_snapshot),
        stage_count=len(workflow.stages),
        artifact_contract_count=len(workflow.artifact_contracts),
        generated_artifacts=list_generated_artifacts(run_dir),
        missing_artifacts=result.missing_artifacts,
        completed_stages=[
            stage.stage_id for stage in stages if stage.status == WorkflowStageStatus.completed
        ],
        failed_stages=[
            stage.stage_id for stage in stages if stage.status == WorkflowStageStatus.failed
        ],
        skipped_stages=[
            stage.stage_id for stage in stages if stage.status == WorkflowStageStatus.skipped
        ],
        degraded_stages=[
            stage.stage_id for stage in stages if stage.status == WorkflowStageStatus.degraded
        ],
        quality_gate_status=quality_gate_status,
        updated_at=now_iso_utc(),
        warnings=[warning.message for warning in result.warnings],
    )


def write_manifest_artifacts(
    run_dir: Path,
    workflow: CompiledWorkflow,
    result: WorkflowExecutionResult,
) -> WorkflowManifest:
    manifest = build_manifest(workflow, result, run_dir)
    write_json_artifact(run_dir, "workflow_manifest.json", manifest)
    write_markdown_artifact(run_dir, "workflow_manifest.md", render_manifest_markdown(manifest))
    if result.readiness is not None:
        write_json_artifact(run_dir, "workflow_readiness.json", result.readiness)
        write_markdown_artifact(
            run_dir,
            "workflow_readiness.md",
            result.readiness.to_markdown(),
        )
    write_json_artifact(run_dir, "workflow_execution_summary.json", result)
    write_markdown_artifact(
        run_dir, "workflow_execution_summary.md", render_execution_summary(result)
    )
    return manifest


def render_manifest_markdown(manifest: WorkflowManifest) -> str:
    lines = [
        "# Workflow Manifest",
        "",
        f"- Workflow: `{manifest.workflow_id}`",
        f"- Thread: `{manifest.thread_id}`",
        f"- Template: `{manifest.template_id}`",
        f"- Mode: `{manifest.mode.value}`",
        f"- Quality gate: `{manifest.quality_gate_status or 'not_run'}`",
        "",
        "## Stages",
        "",
        f"- Completed: {len(manifest.completed_stages)}",
        f"- Skipped: {len(manifest.skipped_stages)}",
        f"- Failed: {len(manifest.failed_stages)}",
        f"- Degraded: {len(manifest.degraded_stages)}",
        "",
        "## Missing Required Artifacts",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest.missing_artifacts)
    if not manifest.missing_artifacts:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"


def render_execution_summary(result: WorkflowExecutionResult) -> str:
    lines = [
        "# Workflow Execution Summary",
        "",
        f"- Workflow: `{result.workflow_id}`",
        f"- Thread: `{result.thread_id}`",
        f"- Status: `{result.status.value}`",
        f"- Degraded: {result.degraded}",
        "",
        "## Stage Status",
    ]
    for stage in result.stages:
        lines.append(f"- `{stage.stage_id}`: {stage.status.value}")
    if result.readiness is not None:
        lines.extend(
            [
                "",
                "## Readiness",
                "",
                f"- Status: `{result.readiness.status.value}`",
                f"- Usable: {result.readiness.usable}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _fingerprint(payload) -> str:
    return hashlib.sha256(
        json.dumps(model_to_plain(payload), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
