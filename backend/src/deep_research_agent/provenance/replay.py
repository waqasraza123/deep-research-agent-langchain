from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import ensure_thread_dir, list_artifacts, safe_thread_id

from .artifact_manifest import build_artifact_manifest
from .contracts import (
    ArtifactManifest,
    ReplayExecutionStep,
    ReplayExecutionSummary,
    ReplayPlan,
)
from .diff import diff_manifests
from .lineage import PROVENANCE_ARTIFACTS, file_sha256, safe_run_dir

REPLAY_EXECUTION_JSON = "replay_execution.json"
REPLAY_EXECUTION_MD = "replay_execution.md"

DEFAULT_REPLAY_SEED_ARTIFACTS = {
    "plan.md",
    "notes.md",
    "report.md",
    "sources.json",
    "metadata.json",
    "strategy.json",
    "strategy.md",
    "subquestions.json",
    "verification_plan.md",
    "protocol_profile.json",
    "protocol_profile.md",
    "protocol_selection.json",
    "protocol_selection.md",
    "source_graph.json",
    "source_graph.md",
    "source_safety.json",
    "source_safety.md",
    "sanitized_sources.json",
    "trust_boundary_policy.md",
    "prompt_injection_findings.json",
    "source_poisoning_findings.json",
}


def default_replay_thread_id(source_thread_id: str) -> str:
    safe_thread_id(source_thread_id)
    return f"replay-{source_thread_id}-{uuid.uuid4().hex[:8]}"


def prepare_replay_run(
    *,
    runs_dir: Path,
    source_thread_id: str,
    replay_thread_id: str | None = None,
    plan: ReplayPlan | None = None,
    source_manifest: ArtifactManifest | None = None,
    include_artifacts: list[str] | None = None,
    allow_overwrite: bool = False,
    offline_only: bool = True,
) -> ReplayExecutionSummary:
    source_dir = safe_run_dir(runs_dir, source_thread_id)
    replay_id = replay_thread_id or default_replay_thread_id(source_thread_id)
    safe_thread_id(replay_id)
    if replay_id == source_thread_id:
        raise ValueError("replay_thread_id must be different from source_thread_id")

    target_dir = (runs_dir.resolve() / replay_id).resolve()
    root = runs_dir.resolve()
    if root != target_dir and root not in target_dir.parents:
        raise ValueError("replay_thread_id escaped runs_dir")
    if target_dir.exists():
        existing = [p for p in target_dir.rglob("*") if p.is_file()]
        if existing and not allow_overwrite:
            raise FileExistsError(replay_id)
        if allow_overwrite:
            shutil.rmtree(target_dir)
    target_dir = ensure_thread_dir(runs_dir, replay_id)

    artifacts_to_copy = set(include_artifacts or DEFAULT_REPLAY_SEED_ARTIFACTS)
    artifacts_to_copy.update(_source_payload_artifacts(source_dir, source_thread_id))
    if source_manifest is not None:
        artifacts_to_copy.update(
            item.artifact_path
            for item in source_manifest.artifacts
            if item.artifact_path.startswith(("sources/", "sanitized_sources/"))
        )

    copied: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []
    for rel_path in sorted(artifacts_to_copy):
        if not _safe_rel_path(rel_path) or rel_path in PROVENANCE_ARTIFACTS:
            skipped.append(rel_path)
            continue
        source_path = (source_dir / rel_path).resolve()
        if not _is_child(source_dir, source_path) or not source_path.exists():
            skipped.append(rel_path)
            continue
        target_path = (target_dir / rel_path).resolve()
        if not _is_child(target_dir, target_path):
            skipped.append(rel_path)
            warnings.append(f"Skipped unsafe replay artifact path: {rel_path}")
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        elif source_path.suffix.lower() == ".json":
            _copy_json_with_rewritten_thread_paths(
                source_path,
                target_path,
                source_thread_id=source_thread_id,
                replay_thread_id=replay_id,
            )
        else:
            shutil.copy2(source_path, target_path)
        copied.append(rel_path)

    summary = ReplayExecutionSummary(
        source_thread_id=source_thread_id,
        replay_thread_id=replay_id,
        generated_at=_now_iso_utc(),
        status="planned",
        offline_only=offline_only,
        plan=plan,
        steps=[
            ReplayExecutionStep(
                step_id="seed",
                name="Copy replay seed artifacts",
                status="completed",
                offline=True,
                copied_artifacts=copied,
                skipped_artifacts=skipped,
                warnings=warnings,
                metadata={"source_run": source_thread_id, "target_run": replay_id},
            )
        ],
        copied_seed_artifacts=copied,
        warnings=warnings,
        replay_artifacts=[artifact.path for artifact in list_artifacts(runs_dir, replay_id)],
    )
    write_replay_execution(target_dir, summary)
    return summary


def finalize_replay_execution(
    *,
    runs_dir: Path,
    source_thread_id: str,
    replay_thread_id: str,
    baseline_manifest: ArtifactManifest,
    plan: ReplayPlan | None,
    steps: list[ReplayExecutionStep],
    rebuilt_layers: list[str],
    skipped_layers: list[str],
    warnings: list[str],
    offline_only: bool = True,
) -> ReplayExecutionSummary:
    target_dir = safe_run_dir(runs_dir, replay_thread_id)
    replay_manifest = build_artifact_manifest(target_dir, replay_thread_id, None)
    diff = diff_manifests(baseline_manifest, replay_manifest)
    matches, mismatches, missing = compare_expected_hashes(plan, replay_manifest)
    status = "completed"
    if any(step.status == "failed" for step in steps):
        status = "failed"
    elif warnings or mismatches or missing or diff.changed_artifacts:
        status = "completed_with_warnings"

    summary = ReplayExecutionSummary(
        source_thread_id=source_thread_id,
        replay_thread_id=replay_thread_id,
        generated_at=_now_iso_utc(),
        status=status,
        offline_only=offline_only,
        plan=plan,
        steps=steps,
        copied_seed_artifacts=[
            artifact
            for step in steps
            for artifact in step.copied_artifacts
            if step.step_id == "seed"
        ],
        rebuilt_layers=rebuilt_layers,
        skipped_layers=skipped_layers,
        warnings=warnings,
        baseline_diff=diff,
        hash_matches=matches,
        hash_mismatches=mismatches,
        missing_expected_artifacts=missing,
        replay_artifacts=[artifact.path for artifact in list_artifacts(runs_dir, replay_thread_id)],
    )
    write_replay_execution(target_dir, summary)
    return summary


def compare_expected_hashes(
    plan: ReplayPlan | None,
    replay_manifest: ArtifactManifest,
) -> tuple[list[str], dict[str, dict[str, str | None]], list[str]]:
    if plan is None:
        return [], {}, []
    by_path = {artifact.artifact_path: artifact for artifact in replay_manifest.artifacts}
    matches: list[str] = []
    mismatches: dict[str, dict[str, str | None]] = {}
    missing: list[str] = []
    for artifact_path, expected_hash in sorted(plan.expected_output_hashes.items()):
        replay_artifact = by_path.get(artifact_path)
        if replay_artifact is None:
            missing.append(artifact_path)
            continue
        actual_hash = replay_artifact.content_hash
        if actual_hash == expected_hash:
            matches.append(artifact_path)
        else:
            mismatches[artifact_path] = {"expected": expected_hash, "actual": actual_hash}
    return matches, mismatches, missing


def write_replay_execution(thread_dir: Path, summary: ReplayExecutionSummary) -> list[str]:
    payload = _model_to_plain(summary)
    (thread_dir / REPLAY_EXECUTION_JSON).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (thread_dir / REPLAY_EXECUTION_MD).write_text(render_replay_execution_markdown(summary), "utf-8")
    return [REPLAY_EXECUTION_JSON, REPLAY_EXECUTION_MD]


def render_replay_execution_markdown(summary: ReplayExecutionSummary) -> str:
    lines = [
        "# Replay Execution",
        "",
        f"- Source run: `{summary.source_thread_id}`",
        f"- Replay run: `{summary.replay_thread_id}`",
        f"- Generated at: `{summary.generated_at}`",
        f"- Status: `{summary.status}`",
        f"- Offline only: `{summary.offline_only}`",
        "",
        "## Steps",
        "",
    ]
    for step in summary.steps:
        lines.extend(
            [
                f"### {step.step_id}",
                "",
                f"- Name: {step.name}",
                f"- Status: `{step.status}`",
                f"- Offline: `{step.offline}`",
                f"- Copied artifacts: {len(step.copied_artifacts)}",
                f"- Generated artifacts: {len(step.generated_artifacts)}",
            ]
        )
        if step.skipped_artifacts:
            lines.append(f"- Skipped artifacts: {len(step.skipped_artifacts)}")
        if step.error:
            lines.append(f"- Error: {step.error}")
        if step.warnings:
            lines.append("- Warnings: " + "; ".join(step.warnings[:8]))
        lines.append("")

    lines.extend(
        [
            "## Rebuilt Layers",
            "",
            *(f"- `{layer}`" for layer in summary.rebuilt_layers),
            "",
            "## Expected Hash Comparison",
            "",
            f"- Matches: {len(summary.hash_matches)}",
            f"- Mismatches: {len(summary.hash_mismatches)}",
            f"- Missing expected artifacts: {len(summary.missing_expected_artifacts)}",
        ]
    )
    if summary.hash_mismatches:
        lines.extend(["", "### Mismatches", ""])
        for artifact, hashes in sorted(summary.hash_mismatches.items()):
            lines.append(
                f"- `{artifact}`: expected `{hashes.get('expected')}`, actual `{hashes.get('actual')}`"
            )
    if summary.missing_expected_artifacts:
        lines.extend(["", "### Missing", ""])
        lines.extend(f"- `{artifact}`" for artifact in summary.missing_expected_artifacts)
    if summary.baseline_diff is not None:
        lines.extend(
            [
                "",
                "## Baseline Diff",
                "",
                f"- Added artifacts: {len(summary.baseline_diff.added_artifacts)}",
                f"- Removed artifacts: {len(summary.baseline_diff.removed_artifacts)}",
                f"- Changed artifacts: {len(summary.baseline_diff.changed_artifacts)}",
                f"- Unchanged artifacts: {len(summary.baseline_diff.unchanged_artifacts)}",
            ]
        )
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    return "\n".join(lines).rstrip() + "\n"


def artifact_hashes(thread_dir: Path, artifacts: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel_path in artifacts:
        if not _safe_rel_path(rel_path):
            continue
        path = thread_dir / rel_path
        if path.exists() and path.is_file():
            hashes[rel_path] = file_sha256(path)
    return hashes


def _source_payload_artifacts(source_dir: Path, source_thread_id: str) -> set[str]:
    manifest_path = source_dir / "sources.json"
    if not manifest_path.exists() or manifest_path.is_dir():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    artifacts: set[str] = set()
    for item in _walk_json(manifest):
        if not isinstance(item, dict):
            continue
        for key in ("local_path", "raw_local_path", "sanitized_local_path"):
            value = item.get(key)
            if isinstance(value, str):
                rel = _run_local_path_to_rel(value, source_thread_id)
                if rel:
                    artifacts.add(rel)
        safety = item.get("source_safety")
        if isinstance(safety, dict):
            for key in ("raw_local_path", "sanitized_local_path"):
                value = safety.get(key)
                if isinstance(value, str):
                    rel = _run_local_path_to_rel(value, source_thread_id)
                    if rel:
                        artifacts.add(rel)
    return artifacts


def _copy_json_with_rewritten_thread_paths(
    source_path: Path,
    target_path: Path,
    *,
    source_thread_id: str,
    replay_thread_id: str,
) -> None:
    try:
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(source_path, target_path)
        return
    rewritten = _rewrite_thread_paths(
        loaded,
        source_thread_id=source_thread_id,
        replay_thread_id=replay_thread_id,
    )
    target_path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_thread_paths(value: Any, *, source_thread_id: str, replay_thread_id: str) -> Any:
    marker = f"runs/{source_thread_id}/"
    replacement = f"runs/{replay_thread_id}/"
    if isinstance(value, dict):
        return {
            str(key): _rewrite_thread_paths(
                item,
                source_thread_id=source_thread_id,
                replay_thread_id=replay_thread_id,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_thread_paths(
                item,
                source_thread_id=source_thread_id,
                replay_thread_id=replay_thread_id,
            )
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(marker, replacement)
    return value


def _run_local_path_to_rel(local_path: str, source_thread_id: str) -> str | None:
    if not local_path or local_path.startswith("/") or "\\" in local_path:
        return None
    marker = f"runs/{source_thread_id}/"
    rel = local_path.split(marker, 1)[-1] if marker in local_path else local_path
    if not _safe_rel_path(rel):
        return None
    return rel


def _safe_rel_path(rel_path: str) -> bool:
    return (
        bool(rel_path)
        and not rel_path.startswith("/")
        and "\\" not in rel_path
        and ".." not in Path(rel_path).parts
    )


def _is_child(root: Path, path: Path) -> bool:
    root = root.resolve()
    path = path.resolve()
    return root == path or root in path.parents


def _walk_json(value: Any) -> list[Any]:
    out = [value]
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_walk_json(item))
    elif isinstance(value, list):
        for item in value:
            out.extend(_walk_json(item))
    return out


def _model_to_plain(model: Any) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()


def _now_iso_utc() -> str:
    from deep_research_agent.artifacts import now_iso_utc

    return now_iso_utc()
