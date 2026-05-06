from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_registry import build_artifact_registry, write_artifact_registry
from .blueprint import generate_blueprint, write_blueprint_artifacts
from .confidence import calibrate_confidence, write_confidence_artifacts
from .contracts import (
    ConfidenceCalibration,
    CritiqueFinding,
    EvidenceUnit,
    KernelArtifactMetadata,
    KernelRunSummary,
    KernelWarning,
    ResearchBlueprint,
    ResearchClaim,
    ResearchKernelInput,
    ResearchKernelSettings,
    ResearchPass,
    SourceUnit,
    VerificationTask,
    model_to_plain,
    stable_id,
    write_json,
)
from .critique import critique_report, extract_claims, write_critique_artifacts
from .errors import KernelPassError
from .passes import create_pass, mark_completed, mark_failed, mark_running, mark_skipped
from .reasoning_units import build_evidence_units, write_evidence_artifacts
from .request_analyzer import analyze_request
from .source_units import build_source_units, write_source_unit_artifacts
from .summary import build_kernel_summary, write_summary_artifacts
from .verification import (
    generate_verification_tasks,
    run_verification_tasks,
    write_verification_artifacts,
)


class KernelPipelineResult:
    def __init__(
        self,
        *,
        blueprint: ResearchBlueprint,
        passes: list[ResearchPass],
        summary: KernelRunSummary,
        warnings: list[KernelWarning],
        artifacts: list[str],
    ) -> None:
        self.blueprint = blueprint
        self.passes = passes
        self.summary = summary
        self.warnings = warnings
        self.artifacts = artifacts

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint": model_to_plain(self.blueprint),
            "passes": model_to_plain(self.passes),
            "summary": model_to_plain(self.summary),
            "warnings": model_to_plain(self.warnings),
            "artifacts": self.artifacts,
        }


def _passes_for(blueprint: ResearchBlueprint) -> list[ResearchPass]:
    records = [
        create_pass(pass_type, required=True, thread_id=blueprint.thread_id)
        for pass_type in blueprint.required_passes
    ]
    records.extend(
        create_pass(pass_type, required=False, thread_id=blueprint.thread_id)
        for pass_type in blueprint.optional_passes
    )
    for pass_type, reason in blueprint.skipped_passes.items():
        if pass_type not in {
            "request_analysis",
            "blueprint_generation",
            "source_inventory",
            "source_unitization",
            "evidence_unitization",
            "agent_execution",
            "report_critique",
            "claim_verification",
            "confidence_calibration",
            "final_kernel_summary",
        }:
            continue
        records.append(
            mark_skipped(
                create_pass(pass_type, required=False, thread_id=blueprint.thread_id), reason
            )
        )
    return records


def write_pass_artifacts(run_dir: Path, passes: list[ResearchPass]) -> list[str]:
    write_json(run_dir / "kernel_passes.json", {"passes": passes})
    lines = [
        "# Kernel Passes",
        "",
        "| Pass | Required | Status | Outputs |",
        "| --- | --- | --- | --- |",
    ]
    for record in passes:
        outputs = (
            ", ".join(record.output_artifacts)
            or record.skipped_reason
            or record.failure_reason
            or ""
        )
        lines.append(
            f"| `{record.pass_type}` | `{record.required}` | `{record.status}` | {outputs[:180]} |"
        )
    (run_dir / "kernel_passes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["kernel_passes.json", "kernel_passes.md"]


def run_kernel_pipeline(
    run_dir: Path,
    kernel_input: ResearchKernelInput,
    settings: ResearchKernelSettings | None = None,
    *,
    existing_blueprint: ResearchBlueprint | None = None,
) -> KernelPipelineResult:
    settings = settings or ResearchKernelSettings()
    warnings: list[KernelWarning] = []
    artifacts: list[str] = []
    intent = None
    complexity = None
    blueprint = existing_blueprint
    source_units: list[SourceUnit] = []
    evidence_units: list[EvidenceUnit] = []
    claims: list[ResearchClaim] = []
    findings: list[CritiqueFinding] = []
    tasks: list[VerificationTask] = []
    calibrations: list[ConfidenceCalibration] = []
    registry: list[KernelArtifactMetadata] = []

    if blueprint is None:
        intent, complexity, analyzer_warnings = analyze_request(
            kernel_input.question, kernel_input.urls
        )
        warnings.extend(analyzer_warnings)
        blueprint = generate_blueprint(
            kernel_input, intent, complexity, settings, analyzer_warnings
        )
    passes = _passes_for(blueprint)

    def execute(pass_type: str, func):
        record = next(
            (p for p in passes if p.pass_type == pass_type and p.status == "pending"), None
        )
        if record is None:
            return None
        mark_running(record)
        try:
            output = func()
            mark_completed(record, output.get("artifacts", []), output.get("metrics", {}))
            artifacts.extend(output.get("artifacts", []))
            warnings.extend(output.get("warnings", []))
            record.warnings.extend(output.get("warnings", []))
            return output
        except Exception as exc:
            mark_failed(record, f"{type(exc).__name__}: {exc}")
            warning = KernelWarning(
                warning_id=stable_id("warn", "pipeline", pass_type, str(exc)),
                subsystem="pipeline",
                code="pass_failed",
                severity="critical" if record.required else "high",
                message=f"{pass_type} failed: {type(exc).__name__}: {exc}",
                recommended_action="Inspect kernel_error.json and rerun rebuild after fixing artifacts.",
            )
            warnings.append(warning)
            record.warnings.append(warning)
            if record.required:
                write_json(
                    run_dir / "kernel_error.json",
                    {"error": record.failure_reason, "pass": pass_type},
                )
                raise KernelPassError(record.failure_reason or "required pass failed") from exc
            return {"artifacts": [], "warnings": [warning], "metrics": {}}

    execute(
        "request_analysis",
        lambda: {
            "artifacts": [],
            "warnings": warnings,
            "metrics": {"intent": blueprint.intent.label},
        },
    )
    execute(
        "blueprint_generation",
        lambda: {
            "artifacts": write_blueprint_artifacts(run_dir, blueprint),
            "warnings": blueprint.operator_warnings,
            "metrics": {"required_passes": len(blueprint.required_passes)},
        },
    )
    execute(
        "source_inventory",
        lambda: {
            "artifacts": [],
            "warnings": [],
            "metrics": {"sources_json_exists": (run_dir / "sources.json").exists()},
        },
    )

    def source_step():
        nonlocal source_units
        source_units, inventory, step_warnings = build_source_units(run_dir, blueprint, settings)
        return {
            "artifacts": write_source_unit_artifacts(run_dir, source_units, inventory),
            "warnings": step_warnings,
            "metrics": inventory,
        }

    execute("source_unitization", source_step)

    def evidence_step():
        nonlocal evidence_units
        evidence_units, coverage, step_warnings = build_evidence_units(
            run_dir, blueprint, source_units, settings
        )
        return {
            "artifacts": write_evidence_artifacts(run_dir, evidence_units, coverage),
            "warnings": step_warnings,
            "metrics": {"evidence_unit_count": len(evidence_units)},
        }

    execute("evidence_unitization", evidence_step)

    def critique_step():
        nonlocal claims, findings
        claims, claim_warnings = extract_claims(run_dir, blueprint, settings)
        findings, critique_warnings = critique_report(
            run_dir, blueprint, source_units, evidence_units, claims
        )
        step_warnings = [*claim_warnings, *critique_warnings]
        return {
            "artifacts": write_critique_artifacts(run_dir, claims, findings, step_warnings),
            "warnings": step_warnings,
            "metrics": {"claim_count": len(claims), "finding_count": len(findings)},
        }

    execute("report_critique", critique_step)

    def verification_step():
        nonlocal tasks
        tasks = generate_verification_tasks(claims, findings, evidence_units, settings)
        tasks = run_verification_tasks(tasks, claims, evidence_units, source_units)
        return {
            "artifacts": write_verification_artifacts(run_dir, tasks),
            "warnings": [],
            "metrics": {"verification_task_count": len(tasks)},
        }

    execute("claim_verification", verification_step)

    def confidence_step():
        nonlocal calibrations
        calibrations = calibrate_confidence(blueprint, source_units, claims, findings, tasks)
        return {
            "artifacts": write_confidence_artifacts(run_dir, calibrations),
            "warnings": [],
            "metrics": {"confidence_targets": len(calibrations)},
        }

    execute("confidence_calibration", confidence_step)

    def summary_step():
        nonlocal registry
        provisional = build_artifact_registry(run_dir, blueprint)
        reg_artifacts = write_artifact_registry(run_dir, provisional)
        registry = build_artifact_registry(run_dir, blueprint)
        summary = build_kernel_summary(
            blueprint,
            passes,
            source_units,
            len(evidence_units),
            claims,
            findings,
            tasks,
            calibrations,
            registry,
        )
        summary_artifacts = write_summary_artifacts(run_dir, summary, source_units, claims, tasks)
        return {
            "artifacts": [*reg_artifacts, *summary_artifacts],
            "warnings": [],
            "metrics": {"final_confidence": summary.final_confidence.confidence_after},
            "summary": summary,
        }

    final_output = execute("final_kernel_summary", summary_step) or {}
    pass_artifacts = write_pass_artifacts(run_dir, passes)
    artifacts.extend(pass_artifacts)
    summary = final_output.get("summary")
    if summary is None:
        if not calibrations:
            calibrations = calibrate_confidence(blueprint, source_units, claims, findings, tasks)
        registry = build_artifact_registry(run_dir, blueprint)
        summary = build_kernel_summary(
            blueprint,
            passes,
            source_units,
            len(evidence_units),
            claims,
            findings,
            tasks,
            calibrations,
            registry,
        )
        artifacts.extend(write_summary_artifacts(run_dir, summary, source_units, claims, tasks))
    return KernelPipelineResult(
        blueprint=blueprint,
        passes=passes,
        summary=summary,
        warnings=warnings,
        artifacts=sorted(set(artifacts)),
    )
