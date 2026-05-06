from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.settings import Settings

from .contracts import CompiledWorkflow, WorkflowStageExecution, WorkflowStageStatus


@dataclass
class WorkflowExecutionContext:
    compiled_workflow: CompiledWorkflow
    settings: Settings
    service: Any | None = None
    dry_run: bool = False
    run_dir: Path | None = None
    stage_outputs: dict[str, WorkflowStageExecution] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    degraded: bool = False
    quality_gate_runner: Any | None = None

    def __post_init__(self) -> None:
        self.run_dir = self.run_dir or ensure_thread_dir(
            self.settings.runs_dir,
            self.compiled_workflow.thread_id,
        )
        self.available_artifacts = self._scan_artifacts()
        self.mock_mode = self.compiled_workflow.settings_snapshot.get(
            "model_provider"
        ) == "mock" or bool(self.compiled_workflow.settings_snapshot.get("mock_mode"))

    def _scan_artifacts(self) -> set[str]:
        assert self.run_dir is not None
        return {
            str(path.relative_to(self.run_dir)).replace("\\", "/")
            for path in self.run_dir.rglob("*")
            if path.is_file()
        }

    def refresh_artifacts(self) -> None:
        self.available_artifacts = self._scan_artifacts()

    def has_runtime_control(self) -> bool:
        return bool(getattr(self.settings, "runtime_control_enabled", False))

    def has_intelligence_kernel(self) -> bool:
        return bool(getattr(self.settings, "intelligence_kernel_enabled", False))

    def has_agent_control(self) -> bool:
        return bool(getattr(self.settings, "agent_control_enabled", False))

    def has_source_safety(self) -> bool:
        return bool(getattr(self.settings, "source_safety_enabled", False))

    def has_document_intelligence(self) -> bool:
        return bool(getattr(self.settings, "document_intelligence_enabled", False))

    def has_retrieval(self) -> bool:
        return bool(getattr(self.settings, "retrieval_enabled", False))

    def has_source_audit(self) -> bool:
        return bool(getattr(self.settings, "source_audit_enabled", False))

    def has_evidence_extraction(self) -> bool:
        return True

    def has_synthesis(self) -> bool:
        return bool(getattr(self.settings, "synthesis_enabled", False))

    def has_verification(self) -> bool:
        return bool(getattr(self.settings, "verification_enabled", False))

    def has_evaluation(self) -> bool:
        return bool(getattr(self.settings, "evaluation_enabled", False))

    def has_provenance(self) -> bool:
        return bool(getattr(self.settings, "provenance_enabled", False))

    def has_evaluation_lab(self) -> bool:
        return bool(getattr(self.settings, "evaluation_lab_enabled", False))

    def has_quality_gates(self) -> bool:
        return bool(
            getattr(self.settings, "evaluation_lab_enabled", False)
            and getattr(self.settings, "evaluation_lab_gates_enabled", False)
        )

    def read_sources(self) -> list[dict[str, Any]]:
        assert self.run_dir is not None
        path = self.run_dir / "sources.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def skip_result(self, stage, reason: str) -> WorkflowStageExecution:
        return WorkflowStageExecution(
            workflow_id=self.compiled_workflow.workflow_id,
            thread_id=self.compiled_workflow.thread_id,
            stage_id=stage.stage_id,
            stage_type=stage.stage_type,
            status=WorkflowStageStatus.skipped,
            skipped_at=_now(),
            warnings=[reason],
        )


def _now() -> str:
    from .contracts import now_iso_utc

    return now_iso_utc()
