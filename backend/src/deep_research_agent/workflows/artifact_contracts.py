from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import artifact_abs_path, list_artifacts

from .contracts import (
    ArtifactType,
    ArtifactValidationReport,
    ArtifactValidationResult,
    ArtifactValidationStatus,
    WorkflowArtifactContract,
    WorkflowStageStatus,
    WorkflowTemplate,
    model_to_plain,
)
from .templates import BASE_CONTRACTS

PLACEHOLDER_MARKERS = (
    "Agent did not write",
    "(Agent did not write",
    "Placeholder",
    "placeholder-only",
)


def build_contracts_for_template(template: WorkflowTemplate) -> list[WorkflowArtifactContract]:
    seen: set[str] = set()
    contracts: list[WorkflowArtifactContract] = []
    for contract in [*BASE_CONTRACTS, *template.artifact_contracts]:
        if contract.artifact_name in seen:
            continue
        seen.add(contract.artifact_name)
        contracts.append(contract)
    return contracts


def validate_artifact_contracts(
    run_dir: Path,
    contracts: list[WorkflowArtifactContract],
    *,
    stage_statuses: dict[str, WorkflowStageStatus | str] | None = None,
) -> ArtifactValidationReport:
    results = [
        validate_single_artifact(run_dir, contract, stage_statuses=stage_statuses)
        for contract in contracts
    ]
    missing_required = [
        result.artifact_name
        for result in results
        if result.required and result.status == ArtifactValidationStatus.failed
    ]
    warnings = [
        message
        for result in results
        if result.status == ArtifactValidationStatus.warning
        for message in result.messages
    ]
    status = ArtifactValidationStatus.passed
    if missing_required:
        status = ArtifactValidationStatus.failed
    elif warnings:
        status = ArtifactValidationStatus.warning
    return ArtifactValidationReport(
        status=status,
        results=results,
        missing_required=missing_required,
        warnings=warnings,
    )


def validate_single_artifact(
    run_dir: Path,
    contract: WorkflowArtifactContract,
    *,
    stage_statuses: dict[str, WorkflowStageStatus | str] | None = None,
) -> ArtifactValidationResult:
    messages: list[str] = []
    try:
        path = _safe_artifact_path(run_dir, contract.artifact_name)
    except ValueError as e:
        return ArtifactValidationResult(
            artifact_name=contract.artifact_name,
            status=ArtifactValidationStatus.failed,
            required=contract.required,
            exists=False,
            messages=[str(e)],
        )
    exists = path.exists()
    if not exists:
        if _missing_allowed(contract, stage_statuses):
            return ArtifactValidationResult(
                artifact_name=contract.artifact_name,
                status=ArtifactValidationStatus.skipped,
                required=contract.required,
                exists=False,
                messages=["Artifact missing but allowed by contract conditions."],
            )
        status = (
            ArtifactValidationStatus.failed
            if contract.required
            else ArtifactValidationStatus.passed
        )
        return ArtifactValidationResult(
            artifact_name=contract.artifact_name,
            status=status,
            required=contract.required,
            exists=False,
            messages=[
                "Required artifact missing." if contract.required else "Optional artifact missing."
            ],
        )
    size = path.stat().st_size
    status = ArtifactValidationStatus.passed
    if contract.min_size_bytes and size < contract.min_size_bytes:
        messages.append(f"Artifact is smaller than {contract.min_size_bytes} bytes.")
        status = (
            ArtifactValidationStatus.failed
            if contract.required
            else ArtifactValidationStatus.warning
        )
    if contract.max_size_bytes is not None and size > contract.max_size_bytes:
        messages.append(f"Artifact exceeds {contract.max_size_bytes} bytes.")
        status = ArtifactValidationStatus.warning
    if contract.must_be_nonempty and _is_empty_text(path):
        messages.append("Artifact text is empty.")
        status = (
            ArtifactValidationStatus.failed
            if contract.required
            else ArtifactValidationStatus.warning
        )
    if contract.must_parse_as_json or contract.artifact_type == ArtifactType.json:
        parsed = parse_json_artifact_safe(path)
        if parsed is None:
            messages.append("Artifact is not parseable JSON.")
            status = (
                ArtifactValidationStatus.failed
                if contract.required
                else ArtifactValidationStatus.warning
            )
        elif _contains_traversal(parsed):
            messages.append("JSON artifact contains path traversal-like values.")
            status = ArtifactValidationStatus.failed
    if contract.artifact_type == ArtifactType.markdown and _is_placeholder_only(path):
        messages.append("Markdown artifact appears to be placeholder-only.")
        status = (
            ArtifactValidationStatus.warning
            if status == ArtifactValidationStatus.passed
            else status
        )
    return ArtifactValidationResult(
        artifact_name=contract.artifact_name,
        status=status,
        required=contract.required,
        exists=True,
        size_bytes=size,
        messages=messages,
    )


def parse_json_artifact_safe(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_artifact_contract_outputs(
    run_dir: Path,
    contracts: list[WorkflowArtifactContract],
    report: ArtifactValidationReport | None = None,
) -> None:
    (run_dir / "workflow_artifact_contracts.json").write_text(
        json.dumps(model_to_plain(contracts), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "workflow_artifact_contracts.md").write_text(
        render_contracts_markdown(contracts),
        encoding="utf-8",
    )
    if report is not None:
        (run_dir / "workflow_artifact_validation.json").write_text(
            json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "workflow_artifact_validation.md").write_text(
            render_validation_markdown(report),
            encoding="utf-8",
        )


def render_contracts_markdown(contracts: list[WorkflowArtifactContract]) -> str:
    lines = ["# Workflow Artifact Contracts", ""]
    for contract in contracts:
        required = "required" if contract.required else "optional"
        lines.append(
            f"- `{contract.artifact_name}` ({contract.artifact_type.value}, {required}) "
            f"from `{contract.producer_stage or 'unknown'}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_validation_markdown(report: ArtifactValidationReport) -> str:
    lines = ["# Workflow Artifact Validation", "", f"Status: `{report.status.value}`", ""]
    for result in report.results:
        lines.append(
            f"- `{result.artifact_name}`: {result.status.value} ({result.size_bytes} bytes)"
        )
        for message in result.messages:
            lines.append(f"  - {message}")
    return "\n".join(lines).rstrip() + "\n"


def list_generated_artifacts(run_dir: Path) -> list[str]:
    try:
        return [item.path for item in list_artifacts(run_dir.parent, run_dir.name)]
    except Exception:
        return sorted(
            str(path.relative_to(run_dir)).replace("\\", "/")
            for path in run_dir.rglob("*")
            if path.is_file()
        )


def _safe_artifact_path(run_dir: Path, rel_path: str) -> Path:
    if not run_dir.name:
        raise ValueError("Invalid run directory")
    try:
        return artifact_abs_path(run_dir.parent, run_dir.name, rel_path)
    except Exception as e:
        raise ValueError(f"Unsafe artifact path: {rel_path}") from e


def _missing_allowed(
    contract: WorkflowArtifactContract,
    stage_statuses: dict[str, WorkflowStageStatus | str] | None,
) -> bool:
    if not contract.allowed_missing_when:
        return False
    statuses = stage_statuses or {}
    for condition in contract.allowed_missing_when:
        if condition.startswith("stage:"):
            _, stage_id, status = condition.split(":", 2)
            if str(statuses.get(stage_id)) == status:
                return True
    return False


def _is_empty_text(path: Path) -> bool:
    try:
        return not path.read_text(encoding="utf-8").strip()
    except Exception:
        return False


def _is_placeholder_only(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    if not text:
        return True
    lowered = text.lower()
    return len(text) < 120 and any(marker.lower() in lowered for marker in PLACEHOLDER_MARKERS)


def _contains_traversal(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_traversal(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_traversal(item) for item in value)
    if isinstance(value, str):
        return "../" in value or "..\\" in value
    return False
