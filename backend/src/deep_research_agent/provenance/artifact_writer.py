from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_manifest import build_artifact_manifest, write_artifact_manifest
from .contracts import (
    ArtifactDependencyGraph,
    ArtifactManifest,
    ProvenanceRecord,
    ReplayPlan,
    ReproducibilityReport,
)
from .dependency_dag import build_dependency_dag, write_dependency_dag
from .lineage import safe_run_dir
from .replay_plan import build_replay_plan, write_replay_plan
from .reproducibility import build_reproducibility_report, write_reproducibility_report


def refresh_provenance_artifacts(
    runs_dir: Path,
    thread_id: str,
    *,
    run: Any | None = None,
) -> ProvenanceRecord:
    thread_dir = safe_run_dir(runs_dir, thread_id)
    _ensure_manifest_placeholders(thread_dir)

    manifest = build_artifact_manifest(thread_dir, thread_id, run)
    graph = build_dependency_dag(manifest)
    reproducibility = build_reproducibility_report(manifest, run)
    replay = build_replay_plan(manifest, reproducibility)
    write_dependency_dag(thread_dir, graph)
    write_reproducibility_report(thread_dir, reproducibility)
    write_replay_plan(thread_dir, replay)

    final_manifest = build_artifact_manifest(thread_dir, thread_id, run)
    write_artifact_manifest(thread_dir, final_manifest)
    final_graph = build_dependency_dag(final_manifest)
    write_dependency_dag(thread_dir, final_graph)
    return ProvenanceRecord(
        thread_id=thread_id,
        generated_at=final_manifest.generated_at,
        manifest=final_manifest,
        dependency_graph=final_graph,
    )


def read_or_build_manifest(
    runs_dir: Path,
    thread_id: str,
    *,
    run: Any | None = None,
) -> ArtifactManifest:
    thread_dir = safe_run_dir(runs_dir, thread_id)
    path = thread_dir / "artifact_manifest.json"
    if not path.exists():
        return refresh_provenance_artifacts(runs_dir, thread_id, run=run).manifest
    return ArtifactManifest.parse_obj(json.loads(path.read_text(encoding="utf-8")))


def read_or_build_dependency_graph(
    runs_dir: Path,
    thread_id: str,
    *,
    run: Any | None = None,
) -> ArtifactDependencyGraph:
    thread_dir = safe_run_dir(runs_dir, thread_id)
    path = thread_dir / "artifact_dependency_dag.json"
    if not path.exists():
        return refresh_provenance_artifacts(runs_dir, thread_id, run=run).dependency_graph
    return ArtifactDependencyGraph.parse_obj(json.loads(path.read_text(encoding="utf-8")))


def read_or_build_reproducibility(
    runs_dir: Path,
    thread_id: str,
    *,
    run: Any | None = None,
) -> ReproducibilityReport:
    thread_dir = safe_run_dir(runs_dir, thread_id)
    path = thread_dir / "reproducibility_report.json"
    if not path.exists():
        refresh_provenance_artifacts(runs_dir, thread_id, run=run)
    return ReproducibilityReport.parse_obj(json.loads(path.read_text(encoding="utf-8")))


def read_or_build_replay_plan(
    runs_dir: Path,
    thread_id: str,
    *,
    run: Any | None = None,
) -> ReplayPlan:
    thread_dir = safe_run_dir(runs_dir, thread_id)
    path = thread_dir / "replay_plan.json"
    if not path.exists():
        refresh_provenance_artifacts(runs_dir, thread_id, run=run)
    return ReplayPlan.parse_obj(json.loads(path.read_text(encoding="utf-8")))


def _ensure_manifest_placeholders(thread_dir: Path) -> None:
    placeholders = {
        "artifact_manifest.json": "{}\n",
        "artifact_manifest.md": "# Artifact Manifest\n\nPending provenance refresh.\n",
    }
    for name, content in placeholders.items():
        path = thread_dir / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
