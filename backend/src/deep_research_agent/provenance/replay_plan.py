from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc

from .contracts import ArtifactManifest, ReplayPlan, ReproducibilityReport


def build_replay_plan(
    manifest: ArtifactManifest,
    reproducibility: ReproducibilityReport,
) -> ReplayPlan:
    artifact_paths = {artifact.artifact_path for artifact in manifest.artifacts}
    steps: list[dict[str, Any]] = []

    def add(
        step_id: str,
        action: str,
        inputs: list[str],
        outputs: list[str],
        offline: bool,
    ) -> None:
        steps.append(
            {
                "step_id": step_id,
                "action": action,
                "inputs": inputs,
                "outputs": outputs,
                "offline": offline,
            }
        )

    add(
        "input",
        "Use the same input question, URLs, protocol/profile selections, and redacted settings.",
        [],
        ["input.question_and_urls"],
        True,
    )
    if "sources.json" in artifact_paths:
        add(
            "sources",
            "Reuse cached sources when allowed; otherwise refetch URLs and compare source hashes.",
            ["input.question_and_urls"],
            ["sources.json", "source_graph.json"],
            False,
        )
    if "document_chunks.json" in artifact_paths or "documents.json" in artifact_paths:
        add(
            "document_intelligence",
            "Rebuild document profiles, sections, tables, and chunks from captured source text.",
            ["sources.json"],
            ["documents.json", "document_chunks.json"],
            True,
        )
    if "context_packs.json" in artifact_paths:
        add(
            "retrieval",
            "Rebuild retrieval index, retrieval results, and context packs.",
            ["document_chunks.json", "sources.json"],
            ["retrieval_index.json", "context_packs.json"],
            True,
        )
    if "report.md" in artifact_paths:
        add(
            "agent",
            "Rerun the agent/report generation if the model provider is available.",
            ["context_packs.json", "notes.md", "sources.json"],
            ["report.md"],
            False,
        )
    if "verification_report.json" in artifact_paths:
        add(
            "verification",
            "Rebuild evidence and verification outputs from the report and sources.",
            ["report.md", "sources.json", "evidence_ledger.json"],
            ["verification_report.json"],
            True,
        )
    if "quality_score.json" in artifact_paths:
        add(
            "evaluation",
            "Rebuild evaluation and quality-score artifacts.",
            ["verification_report.json", "report.md", "sources.json"],
            ["quality_score.json", "evaluation_report.json"],
            True,
        )
    add(
        "compare",
        "Compare regenerated artifact hashes, sizes, and producer metadata against the manifest.",
        sorted(artifact_paths),
        ["artifact_diff_summary"],
        True,
    )

    return ReplayPlan(
        thread_id=manifest.thread_id,
        generated_at=now_iso_utc(),
        ordered_steps=steps,
        required_artifacts=sorted(
            path
            for path in ("sources.json", "report.md", "artifact_manifest.json")
            if path in artifact_paths
        ),
        optional_artifacts=sorted(
            path for path in artifact_paths if path not in {"sources.json", "report.md"}
        ),
        expected_output_hashes={
            artifact.artifact_path: artifact.content_hash
            for artifact in manifest.artifacts
            if artifact.content_hash != "self-referential"
        },
        warnings=reproducibility.not_reproducible,
    )


def write_replay_plan(thread_dir: Path, plan: ReplayPlan) -> list[str]:
    payload = _model_to_plain(plan)
    (thread_dir / "replay_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "replay_plan.md").write_text(render_replay_plan_markdown(plan), encoding="utf-8")
    return ["replay_plan.json", "replay_plan.md"]


def render_replay_plan_markdown(plan: ReplayPlan) -> str:
    lines = [
        "# Replay Plan",
        "",
        f"- Thread ID: `{plan.thread_id}`",
        f"- Generated at: `{plan.generated_at}`",
        f"- Steps: {len(plan.ordered_steps)}",
        "",
        "## Ordered Steps",
        "",
    ]
    for idx, step in enumerate(plan.ordered_steps, start=1):
        offline = "offline" if step.get("offline") else "requires provider/live access"
        lines.extend(
            [
                f"### {idx}. {step['step_id']}",
                "",
                f"- Action: {step['action']}",
                f"- Inputs: {', '.join(f'`{item}`' for item in step.get('inputs', [])) or 'none'}",
                "- Outputs: "
                f"{', '.join(f'`{item}`' for item in step.get('outputs', [])) or 'none'}",
                f"- Mode: {offline}",
                "",
            ]
        )
    lines.extend(["## Expected Output Hashes", ""])
    for artifact, content_hash in sorted(plan.expected_output_hashes.items()):
        lines.append(f"- `{artifact}`: `{content_hash}`")
    return "\n".join(lines).rstrip() + "\n"


def _model_to_plain(model: Any) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
