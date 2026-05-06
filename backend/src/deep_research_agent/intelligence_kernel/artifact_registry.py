from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import (
    KernelArtifactMetadata,
    KernelWarning,
    ResearchBlueprint,
    stable_id,
    write_json,
)


def _safe_rel(run_dir: Path, path: Path) -> str:
    rel = path.resolve().relative_to(run_dir.resolve())
    rel_s = str(rel).replace("\\", "/")
    if rel_s.startswith("/") or ".." in rel_s.split("/"):
        raise ValueError("Unsafe artifact path")
    return rel_s


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_type(name: str) -> str:
    if name.endswith(".json"):
        return "json"
    if name.endswith(".md"):
        return "markdown"
    if name.endswith(".txt"):
        return "text"
    return "artifact"


def build_artifact_registry(
    run_dir: Path, blueprint: ResearchBlueprint
) -> list[KernelArtifactMetadata]:
    expected = set(blueprint.expected_artifacts)
    discovered: dict[str, KernelArtifactMetadata] = {}
    for path in run_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = _safe_rel(run_dir, path)
        discovered[rel] = KernelArtifactMetadata(
            name=Path(rel).name,
            path=rel,
            type=_artifact_type(rel),
            size_bytes=path.stat().st_size,
            content_hash=_hash(path),
            producer="intelligence_kernel"
            if rel in expected or rel.startswith("kernel_")
            else "run",
            required=rel in expected,
            exists=True,
        )
    for rel in sorted(expected - set(discovered)):
        discovered[rel] = KernelArtifactMetadata(
            name=Path(rel).name,
            path=rel,
            type=_artifact_type(rel),
            producer="intelligence_kernel",
            required=True,
            exists=False,
            warnings=[
                KernelWarning(
                    warning_id=stable_id("warn", "artifact_registry", rel),
                    subsystem="artifact_registry",
                    code="expected_artifact_missing",
                    severity="medium",
                    message=f"Expected artifact {rel} is missing.",
                    affected_artifacts=[rel],
                    recommended_action="Rerun the intelligence rebuild.",
                )
            ],
        )
    return [discovered[key] for key in sorted(discovered)]


def render_registry_markdown(items: list[KernelArtifactMetadata]) -> str:
    lines = [
        "# Kernel Artifact Registry",
        "",
        "| Path | Type | Required | Exists | Size | Hash |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in items:
        lines.append(
            f"| `{item.path}` | `{item.type}` | `{item.required}` | `{item.exists}` | {item.size_bytes} | `{(item.content_hash or '')[:12]}` |"
        )
    return "\n".join(lines) + "\n"


def write_artifact_registry(run_dir: Path, items: list[KernelArtifactMetadata]) -> list[str]:
    write_json(run_dir / "kernel_artifact_registry.json", {"artifacts": items})
    (run_dir / "kernel_artifact_registry.md").write_text(
        render_registry_markdown(items), encoding="utf-8"
    )
    return ["kernel_artifact_registry.json", "kernel_artifact_registry.md"]
