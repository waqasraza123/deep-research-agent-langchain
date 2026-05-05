from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import RunStatus
from .repository import RunRepository
from .state_machine import TERMINAL_STATUSES


class CleanupPlanItem(BaseModel):
    thread_id: str
    reason: str
    size_bytes: int
    created_at: datetime | None = None
    status: RunStatus | None = None
    paths: list[str] = Field(default_factory=list)


class CleanupPlan(BaseModel):
    generated_at: datetime
    stale_after_hours: int
    max_dir_bytes: int
    delete_requested: bool = False
    items: list[CleanupPlanItem] = Field(default_factory=list)


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def build_cleanup_plan(
    repository: RunRepository,
    *,
    stale_after_hours: int = 24,
    max_dir_bytes: int = 50_000_000,
) -> CleanupPlan:
    now = datetime.now(timezone.utc)
    stale_before = now - timedelta(hours=stale_after_hours)
    items: dict[str, CleanupPlanItem] = {}

    for run in repository.list():
        run_dir = repository.runs_dir / run.thread_id
        size = directory_size(run_dir)
        reasons: list[str] = []
        if run.status not in TERMINAL_STATUSES and run.updated_at < stale_before:
            reasons.append("stale incomplete run")
        if size > max_dir_bytes:
            reasons.append("artifact directory exceeds size limit")
        if not reasons:
            continue
        items[run.thread_id] = CleanupPlanItem(
            thread_id=run.thread_id,
            reason=", ".join(reasons),
            size_bytes=size,
            created_at=run.created_at,
            status=run.status,
            paths=[str(run_dir)],
        )

    return CleanupPlan(
        generated_at=now,
        stale_after_hours=stale_after_hours,
        max_dir_bytes=max_dir_bytes,
        items=list(items.values()),
    )


def apply_cleanup_plan(
    repository: RunRepository,
    plan: CleanupPlan,
    *,
    thread_ids: list[str],
    confirm_delete: bool,
) -> CleanupPlan:
    if not confirm_delete:
        raise ValueError("confirm_delete must be true to delete run artifacts")

    planned = {item.thread_id for item in plan.items}
    for thread_id in thread_ids:
        if thread_id in planned:
            repository.delete_run_dir(thread_id)
    plan.delete_requested = True
    return plan
