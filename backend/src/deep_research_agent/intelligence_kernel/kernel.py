from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..artifacts import ensure_required_artifacts
from .contracts import ResearchKernelInput, ResearchKernelSettings, model_to_plain
from .pipeline import KernelPipelineResult, run_kernel_pipeline


def settings_from_runtime(settings: Any) -> ResearchKernelSettings:
    return ResearchKernelSettings(
        intelligence_kernel_enabled=bool(getattr(settings, "intelligence_kernel_enabled", True)),
        offline_mode=bool(getattr(settings, "intelligence_offline_mode", True)),
        mock_model_allowed=bool(getattr(settings, "intelligence_mock_model_allowed", True)),
        source_reasoning_enabled=bool(
            getattr(settings, "intelligence_source_reasoning_enabled", True)
        ),
        critique_enabled=bool(getattr(settings, "intelligence_critique_enabled", True)),
        verification_enabled=bool(getattr(settings, "intelligence_verification_enabled", True)),
        confidence_enabled=bool(getattr(settings, "intelligence_confidence_enabled", True)),
        max_reasoning_passes=int(getattr(settings, "intelligence_max_reasoning_passes", 8)),
        max_source_units=int(getattr(settings, "intelligence_max_source_units", 100)),
        max_evidence_units=int(getattr(settings, "intelligence_max_evidence_units", 500)),
        max_claims=int(getattr(settings, "intelligence_max_claims", 200)),
        max_claims_to_verify=int(getattr(settings, "intelligence_max_claims_to_verify", 50)),
        max_artifact_bytes=int(getattr(settings, "intelligence_max_artifact_bytes", 5_000_000)),
        strict_citation_mode=bool(getattr(settings, "intelligence_strict_citation_mode", False)),
        sensitive_domain_review_required=bool(
            getattr(settings, "intelligence_sensitive_domain_review_required", True)
        ),
        fail_on_critical_warnings=bool(
            getattr(settings, "intelligence_fail_on_critical_warnings", False)
        ),
        produce_markdown_artifacts=bool(
            getattr(settings, "intelligence_produce_markdown_artifacts", True)
        ),
        produce_json_artifacts=bool(getattr(settings, "intelligence_produce_json_artifacts", True)),
    )


def snapshot_settings(settings: Any) -> dict[str, Any]:
    if settings is None:
        return {}
    if is_dataclass(settings):
        data = asdict(settings)
    elif hasattr(settings, "model_dump"):
        data = settings.model_dump(mode="json")
    elif hasattr(settings, "dict"):
        data = settings.dict()
    else:
        data = dict(getattr(settings, "__dict__", {}))
    return {
        k: str(v) if isinstance(v, Path) else v
        for k, v in data.items()
        if "key" not in k.lower() and "secret" not in k.lower()
    }


def build_kernel_input(
    *,
    thread_id: str,
    question: str,
    urls: list[str] | None = None,
    settings: Any = None,
    requested_outputs: list[str] | None = None,
    raw_request_metadata: dict[str, Any] | None = None,
) -> ResearchKernelInput:
    return ResearchKernelInput(
        thread_id=thread_id,
        question=question,
        urls=urls or [],
        settings_snapshot=snapshot_settings(settings),
        requested_outputs=requested_outputs or [],
        raw_request_metadata=raw_request_metadata or {},
    )


def rebuild_intelligence_kernel(
    *,
    runs_dir: Path,
    thread_id: str,
    question: str,
    urls: list[str] | None = None,
    runtime_settings: Any = None,
    raw_request_metadata: dict[str, Any] | None = None,
) -> KernelPipelineResult:
    warnings = ensure_required_artifacts(runs_dir, thread_id)
    run_dir = (runs_dir / thread_id).resolve()
    kernel_settings = settings_from_runtime(runtime_settings)
    kernel_input = build_kernel_input(
        thread_id=thread_id,
        question=question,
        urls=urls or [],
        settings=runtime_settings,
        raw_request_metadata={**(raw_request_metadata or {}), "backfill_warnings": warnings},
    )
    return run_kernel_pipeline(run_dir, kernel_input, kernel_settings)


def read_kernel_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "kernel_summary.json"
    if not path.exists():
        return {
            "thread_id": run_dir.name,
            "status": "missing",
            "message": "kernel_summary.json is missing; rebuild intelligence for this run.",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def result_to_plain(result: KernelPipelineResult) -> dict[str, Any]:
    return model_to_plain(result.to_dict())
