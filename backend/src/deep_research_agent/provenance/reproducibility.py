from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc

from .contracts import ArtifactManifest, ReproducibilityReport
from .lineage import redact_secrets, settings_snapshot_from_run


def build_reproducibility_report(
    manifest: ArtifactManifest,
    run: Any | None = None,
) -> ReproducibilityReport:
    settings_used = redact_secrets(settings_snapshot_from_run(run))
    live_sources = [source for source in manifest.sources if source.live_dependency]
    nondeterministic_models = [
        model for model in manifest.model_invocations if model.nondeterministic
    ]
    credentials = sorted(
        {
            credential
            for model in manifest.model_invocations
            for credential in model.credentials_required
        }
    )
    providers = sorted({model.provider for model in manifest.model_invocations if model.provider})

    live_source_artifacts = sorted(
        {
            artifact.artifact_path
            for artifact in manifest.artifacts
            if artifact.source_dependencies and live_sources
        }
    )
    model_artifacts = sorted(
        {
            artifact.artifact_path
            for artifact in manifest.artifacts
            if artifact.model_dependencies and nondeterministic_models
        }
    )
    offline_artifacts = sorted(
        artifact.artifact_path
        for artifact in manifest.artifacts
        if not artifact.model_dependencies
        and not artifact.source_dependencies
        and artifact.producer_subsystem != "provenance"
    )
    not_reproducible: list[str] = []
    if live_sources:
        not_reproducible.append(
            "Live source bytes may differ unless cached source artifacts are reused."
        )
    if nondeterministic_models:
        not_reproducible.append(
            "Model generations may differ even with the same prompt, settings, and provider."
        )
    if credentials:
        not_reproducible.append(
            "Provider credentials are required to rerun model-dependent stages."
        )
    can_replay = bool(manifest.artifacts)
    if not can_replay:
        status = "not_replayable"
    elif live_sources or nondeterministic_models:
        status = "partially_replayable"
    else:
        status = "replayable"
    return ReproducibilityReport(
        thread_id=manifest.thread_id,
        generated_at=now_iso_utc(),
        status=status,
        can_replay=can_replay,
        replayable_offline_artifacts=offline_artifacts,
        live_source_dependent_artifacts=live_source_artifacts,
        model_nondeterministic_artifacts=model_artifacts,
        sources_may_have_changed=live_sources,
        credentials_required=credentials,
        providers_required=providers,
        settings_used=settings_used,
        not_reproducible=not_reproducible,
        warnings=manifest.warnings,
    )


def write_reproducibility_report(
    thread_dir: Path,
    report: ReproducibilityReport,
) -> list[str]:
    payload = _model_to_plain(report)
    (thread_dir / "reproducibility_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "reproducibility_report.md").write_text(
        render_reproducibility_report_markdown(report),
        encoding="utf-8",
    )
    return ["reproducibility_report.json", "reproducibility_report.md"]


def render_reproducibility_report_markdown(report: ReproducibilityReport) -> str:
    lines = [
        "# Reproducibility Report",
        "",
        f"- Thread ID: `{report.thread_id}`",
        f"- Generated at: `{report.generated_at}`",
        f"- Status: `{report.status}`",
        f"- Can replay: `{report.can_replay}`",
        f"- Providers needed: {', '.join(report.providers_required) or 'none'}",
        f"- Credentials needed: {', '.join(report.credentials_required) or 'none'}",
        "",
        "## External Live Source Dependencies",
        "",
    ]
    if report.sources_may_have_changed:
        for source in report.sources_may_have_changed:
            lines.append(f"- `{source.source_id}` {source.url}")
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Model Nondeterminism", ""])
    if report.model_nondeterministic_artifacts:
        for artifact in report.model_nondeterministic_artifacts:
            lines.append(f"- `{artifact}`")
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Offline-Regenerable Artifacts", ""])
    if report.replayable_offline_artifacts:
        for artifact in report.replayable_offline_artifacts:
            lines.append(f"- `{artifact}`")
    else:
        lines.append("- None identified.")
    lines.extend(["", "## Not Reproducible Without Capture", ""])
    if report.not_reproducible:
        lines.extend(f"- {item}" for item in report.not_reproducible)
    else:
        lines.append("- No limitations detected.")
    lines.extend(["", "## Settings Used", "", "```json"])
    lines.append(json.dumps(report.settings_used, indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines).rstrip() + "\n"


def _model_to_plain(model: Any) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
