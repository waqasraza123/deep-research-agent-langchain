from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PIPELINE_SUMMARY_ARTIFACTS = (
    "intelligence_pipeline_summary.json",
    "intelligence_pipeline_summary.md",
)


class PipelineStageSummary(BaseModel):
    stage: str
    enabled: bool = True
    artifact_paths: list[str] = Field(default_factory=list)
    status: str = "unknown"
    counts: dict[str, int | float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class IntelligencePipelineSummary(BaseModel):
    thread_id: str
    generated_at: str
    question: str = ""
    protocol_id: str | None = None
    intelligence_profile_id: str | None = None
    source_count: int = 0
    discovered_source_count: int = 0
    document_count: int = 0
    chunk_count: int = 0
    retrieval_result_count: int = 0
    verification_task_count: int = 0
    confidence_after: float | None = None
    review_recommended: bool = False
    stages: list[PipelineStageSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Any:
    try:
        if not path.exists() or path.is_dir():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_paths(run_dir: Path, names: list[str]) -> list[str]:
    return [name for name in names if (run_dir / name).exists()]


def build_intelligence_pipeline_summary(
    run_dir: Path,
    *,
    thread_id: str,
    question: str = "",
    confidence_threshold_for_review: float = 0.55,
) -> IntelligencePipelineSummary:
    warnings: list[str] = []
    protocol = _read_json(run_dir / "protocol_selection.json")
    sources = _read_json(run_dir / "sources.json")
    discovery_selection = _read_json(run_dir / "source_selection.json")
    documents = _read_json(run_dir / "document_profiles.json")
    retrieval_index = _read_json(run_dir / "retrieval_index.json")
    retrieval_results = _read_json(run_dir / "retrieval_results.json")
    context_packs = _read_json(run_dir / "context_packs.json")
    verification = _read_json(run_dir / "verification_results.json")
    confidence = _read_json(run_dir / "confidence_calibration.json")

    protocol_id = None
    profile_id = None
    review_recommended = False
    if isinstance(protocol, dict):
        selected = protocol.get("selected_protocol") or {}
        profile = protocol.get("intelligence_profile") or {}
        protocol_id = selected.get("protocol_id")
        profile_id = profile.get("profile_id")
        review_recommended = bool(protocol.get("review_recommended"))

    source_items = sources if isinstance(sources, list) else []
    discovered_count = len(
        [
            item
            for item in source_items
            if isinstance(item, dict) and item.get("source_kind") == "auto_discovered"
        ]
    )
    selected_discovery = []
    if isinstance(discovery_selection, dict):
        selected_discovery = discovery_selection.get("selected_candidates") or []

    document_profiles = []
    if isinstance(documents, dict):
        document_profiles = documents.get("profiles") or []
    document_chunk_count = sum(
        len(profile.get("chunks") or [])
        for profile in document_profiles
        if isinstance(profile, dict)
    )
    retrieval_chunks = []
    if isinstance(retrieval_index, dict):
        retrieval_chunks = retrieval_index.get("chunks") or []
    retrieval_result_items = []
    if isinstance(retrieval_results, dict):
        retrieval_result_items = retrieval_results.get("results") or []

    verification_summary = verification.get("summary") if isinstance(verification, dict) else {}
    verification_task_count = int((verification_summary or {}).get("total_tasks") or 0)
    confidence_after = None
    if isinstance(confidence, dict):
        raw_confidence = confidence.get("report_confidence_after")
        try:
            confidence_after = float(raw_confidence)
        except Exception:
            confidence_after = None
        if (
            confidence_after is not None
            and confidence_after < confidence_threshold_for_review
        ):
            review_recommended = True

    stages = [
        PipelineStageSummary(
            stage="protocol_selection",
            artifact_paths=_artifact_paths(
                run_dir,
                [
                    "protocol_selection.json",
                    "protocol_selection.md",
                    "intelligence_profile.json",
                    "protocol_instructions.md",
                    "policy_requirements.json",
                    "policy_warnings.md",
                ],
            ),
            status="completed" if protocol else "missing",
            counts={
                "warnings": len(protocol.get("warnings", []))
                if isinstance(protocol, dict)
                else 0
            },
        ),
        PipelineStageSummary(
            stage="source_discovery",
            artifact_paths=_artifact_paths(
                run_dir,
                [
                    "source_acquisition_plan.json",
                    "search_queries.json",
                    "source_candidates.json",
                    "source_selection.json",
                    "source_discovery_summary.md",
                ],
            ),
            status="completed"
            if (run_dir / "source_acquisition_plan.json").exists()
            else "missing",
            counts={"selected": len(selected_discovery), "auto_discovered": discovered_count},
        ),
        PipelineStageSummary(
            stage="document_intelligence",
            artifact_paths=_artifact_paths(
                run_dir,
                [
                    "document_profiles.json",
                    "document_chunks.jsonl",
                    "document_tables.json",
                    "document_citations.json",
                    "document_warnings.md",
                ],
            ),
            status="completed" if documents is not None else "missing",
            counts={"documents": len(document_profiles), "chunks": document_chunk_count},
        ),
        PipelineStageSummary(
            stage="retrieval",
            artifact_paths=_artifact_paths(
                run_dir,
                [
                    "retrieval_index.json",
                    "retrieval_queries.json",
                    "retrieval_results.json",
                    "context_packs.json",
                    "context_packs.md",
                    "retrieval_coverage.md",
                ],
            ),
            status="completed" if retrieval_index is not None else "missing",
            counts={
                "chunks": len(retrieval_chunks),
                "results": len(retrieval_result_items),
                "packs": len((context_packs or {}).get("packs", {}))
                if isinstance(context_packs, dict)
                else 0,
            },
            warnings=list(retrieval_index.get("warnings", []))
            if isinstance(retrieval_index, dict)
            else [],
        ),
        PipelineStageSummary(
            stage="verification",
            artifact_paths=_artifact_paths(
                run_dir,
                [
                    "verification_plan.json",
                    "verification_tasks.json",
                    "verification_results.json",
                    "verification_report.md",
                    "confidence_calibration.json",
                    "claim_rewrite_suggestions.md",
                ],
            ),
            status="completed" if verification is not None else "missing",
            counts={
                "tasks": verification_task_count,
                "high_priority_open": int(
                    (verification_summary or {}).get("high_priority_open_issues") or 0
                ),
            },
            warnings=list((verification_summary or {}).get("warnings", [])),
        ),
    ]
    for stage in stages:
        warnings.extend(stage.warnings)

    return IntelligencePipelineSummary(
        thread_id=thread_id,
        generated_at=_now_iso_utc(),
        question=question,
        protocol_id=protocol_id,
        intelligence_profile_id=profile_id,
        source_count=len(source_items),
        discovered_source_count=discovered_count,
        document_count=len(document_profiles),
        chunk_count=max(document_chunk_count, len(retrieval_chunks)),
        retrieval_result_count=len(retrieval_result_items),
        verification_task_count=verification_task_count,
        confidence_after=confidence_after,
        review_recommended=review_recommended,
        stages=stages,
        warnings=list(dict.fromkeys(warnings)),
    )


def write_intelligence_pipeline_summary(
    run_dir: Path,
    summary: IntelligencePipelineSummary,
) -> list[str]:
    payload = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else summary.dict()
    (run_dir / "intelligence_pipeline_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "intelligence_pipeline_summary.md").write_text(
        render_intelligence_pipeline_summary_markdown(summary),
        encoding="utf-8",
    )
    return list(PIPELINE_SUMMARY_ARTIFACTS)


def render_intelligence_pipeline_summary_markdown(
    summary: IntelligencePipelineSummary,
) -> str:
    lines = [
        "# Intelligence Pipeline Summary",
        "",
        f"- Thread: `{summary.thread_id}`",
        f"- Generated: `{summary.generated_at}`",
        f"- Protocol: `{summary.protocol_id or 'unknown'}`",
        f"- Profile: `{summary.intelligence_profile_id or 'unknown'}`",
        f"- Sources: {summary.source_count} ({summary.discovered_source_count} auto-discovered)",
        f"- Documents: {summary.document_count}",
        f"- Chunks: {summary.chunk_count}",
        f"- Retrieval results: {summary.retrieval_result_count}",
        f"- Verification tasks: {summary.verification_task_count}",
        f"- Confidence after verification: `{summary.confidence_after}`",
        f"- Review recommended: `{summary.review_recommended}`",
        "",
        "## Stages",
        "",
    ]
    for stage in summary.stages:
        lines.extend(
            [
                f"### {stage.stage}",
                "",
                f"- Status: `{stage.status}`",
                f"- Artifacts: {', '.join(f'`{path}`' for path in stage.artifact_paths) or 'none'}",
            ]
        )
        if stage.counts:
            lines.append(
                "- Counts: "
                + ", ".join(f"{key}={value}" for key, value in stage.counts.items())
            )
        if stage.warnings:
            lines.append("- Warnings: " + "; ".join(stage.warnings[:5]))
        lines.append("")
    if summary.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    return "\n".join(lines).rstrip() + "\n"
