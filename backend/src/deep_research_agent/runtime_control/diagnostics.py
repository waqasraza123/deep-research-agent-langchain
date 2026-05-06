from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ResearchJobStatus, RuntimeDiagnostics
from .idempotency import redact_settings_snapshot
from .repository import RuntimeRepository


def build_control_diagnostics(
    *,
    repository: RuntimeRepository,
    runs_dir: Path,
    settings: Any,
) -> RuntimeDiagnostics:
    counts = repository.status_counts()
    expired = repository.list_expired_leases()
    warnings: list[str] = []
    if not runs_dir.exists():
        warnings.append("Runs directory does not exist.")
    return RuntimeDiagnostics(
        runtime_enabled=bool(getattr(settings, "runtime_control_enabled", True)),
        queue_backend="sqlite",
        repository_backend="sqlite",
        queued_jobs=counts.get(ResearchJobStatus.QUEUED.value, 0),
        running_jobs=counts.get(ResearchJobStatus.RUNNING.value, 0)
        + counts.get(ResearchJobStatus.LEASED.value, 0),
        paused_jobs=counts.get(ResearchJobStatus.PAUSED.value, 0),
        failed_jobs=counts.get(ResearchJobStatus.FAILED.value, 0),
        dead_lettered_jobs=counts.get(ResearchJobStatus.DEAD_LETTERED.value, 0),
        completed_jobs=counts.get(ResearchJobStatus.COMPLETED.value, 0),
        expired_leases=len(expired),
        stale_running_jobs=len(expired),
        warnings=warnings,
        sqlite_path=str(repository.db_path),
        settings_summary=redact_settings_snapshot(settings),
    )

