from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_manifest import build_artifact_manifest
from .contracts import ArtifactDiffSummary, ArtifactManifest
from .lineage import safe_run_dir


def diff_run_dirs(
    runs_dir: Path,
    left_thread_id: str,
    right_thread_id: str,
    *,
    left_run: Any | None = None,
    right_run: Any | None = None,
) -> ArtifactDiffSummary:
    left_dir = safe_run_dir(runs_dir, left_thread_id)
    right_dir = safe_run_dir(runs_dir, right_thread_id)
    return diff_manifests(
        build_artifact_manifest(left_dir, left_thread_id, left_run),
        build_artifact_manifest(right_dir, right_thread_id, right_run),
    )


def diff_manifest_files(left_path: Path, right_path: Path) -> ArtifactDiffSummary:
    return diff_manifests(read_manifest_file(left_path), read_manifest_file(right_path))


def read_manifest_file(path: Path) -> ArtifactManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ArtifactManifest.parse_obj(data)


def diff_manifests(
    left: ArtifactManifest,
    right: ArtifactManifest,
) -> ArtifactDiffSummary:
    left_by_path = {item.artifact_path: item for item in left.artifacts}
    right_by_path = {item.artifact_path: item for item in right.artifacts}
    left_paths = set(left_by_path)
    right_paths = set(right_by_path)

    added = sorted(right_paths - left_paths)
    removed = sorted(left_paths - right_paths)
    unchanged: list[str] = []
    changed: list[str] = []
    changed_hashes: dict[str, dict[str, str | None]] = {}
    changed_sizes: dict[str, dict[str, int | None]] = {}
    changed_producers: dict[str, dict[str, str | None]] = {}

    for path in sorted(left_paths & right_paths):
        left_item = left_by_path[path]
        right_item = right_by_path[path]
        is_changed = False
        if left_item.content_hash != right_item.content_hash:
            is_changed = True
            changed_hashes[path] = {
                "left": left_item.content_hash,
                "right": right_item.content_hash,
            }
        if left_item.size_bytes != right_item.size_bytes:
            is_changed = True
            changed_sizes[path] = {"left": left_item.size_bytes, "right": right_item.size_bytes}
        if left_item.producer_subsystem != right_item.producer_subsystem:
            is_changed = True
            changed_producers[path] = {
                "left": left_item.producer_subsystem,
                "right": right_item.producer_subsystem,
            }
        if is_changed:
            changed.append(path)
        else:
            unchanged.append(path)

    return ArtifactDiffSummary(
        added_artifacts=added,
        removed_artifacts=removed,
        changed_artifacts=changed,
        unchanged_artifacts=unchanged,
        changed_hashes=changed_hashes,
        changed_sizes=changed_sizes,
        changed_producer_subsystems=changed_producers,
    )
