from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc, safe_thread_id

RetentionClass = Literal["standard", "audit", "regulated", "ephemeral"]

RETENTION_POLICY_JSON = "retention_policy.json"
RETENTION_POLICY_MD = "retention_policy.md"

DEFAULT_RETENTION_DAYS: dict[RetentionClass, int] = {
    "ephemeral": 7,
    "standard": 30,
    "audit": 365,
    "regulated": 2555,
}


class RetentionHold(BaseModel):
    hold_id: str
    reason: str
    requested_by: str = "operator"
    created_at: str = Field(default_factory=now_iso_utc)
    released_at: str | None = None
    released_by: str | None = None
    release_reason: str = ""

    @property
    def active(self) -> bool:
        return self.released_at is None


class RetentionPolicy(BaseModel):
    policy_version: str = "1.0"
    thread_id: str
    generated_at: str = Field(default_factory=now_iso_utc)
    retention_class: RetentionClass = "standard"
    retain_until: str | None = None
    delete_after: str | None = None
    legal_hold: bool = False
    holds: list[RetentionHold] = Field(default_factory=list)
    reason: str = ""
    requested_by: str = "system"
    warnings: list[str] = Field(default_factory=list)

    @property
    def active_holds(self) -> list[RetentionHold]:
        return [hold for hold in self.holds if hold.active]


class RetentionPolicyRequest(BaseModel):
    retention_class: RetentionClass = "standard"
    retain_days: int | None = Field(default=None, ge=0)
    retain_until: str | None = None
    delete_after: str | None = None
    legal_hold: bool = False
    reason: str = ""
    requested_by: str = "operator"


class RetentionHoldRequest(BaseModel):
    hold_id: str | None = None
    reason: str
    requested_by: str = "operator"


class RetentionReleaseRequest(BaseModel):
    hold_id: str | None = None
    released_by: str = "operator"
    reason: str = ""


def build_retention_policy(
    *,
    runs_dir: Path,
    thread_id: str,
    request: RetentionPolicyRequest,
    existing: RetentionPolicy | None = None,
) -> RetentionPolicy:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    now = datetime.now(timezone.utc)
    retain_until = _resolve_retain_until(now, request)
    delete_after = _parse_iso_datetime(request.delete_after) if request.delete_after else None
    warnings: list[str] = []
    if delete_after is not None and retain_until is not None and delete_after < retain_until:
        warnings.append("delete_after is earlier than retain_until; cleanup must honor retain_until.")
    holds = list(existing.holds if existing is not None else [])
    policy = RetentionPolicy(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        retention_class=request.retention_class,
        retain_until=_format_dt(retain_until),
        delete_after=_format_dt(delete_after),
        legal_hold=bool(request.legal_hold or any(hold.active for hold in holds)),
        holds=holds,
        reason=request.reason,
        requested_by=request.requested_by,
        warnings=warnings,
    )
    write_retention_policy(run_dir, policy)
    return policy


def read_retention_policy(runs_dir: Path, thread_id: str) -> RetentionPolicy:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    path = run_dir / RETENTION_POLICY_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(RetentionPolicy, "model_validate", None)
    if callable(validate):
        return validate(data)
    return RetentionPolicy.parse_obj(data)


def read_retention_policy_or_none(runs_dir: Path, thread_id: str) -> RetentionPolicy | None:
    try:
        return read_retention_policy(runs_dir, thread_id)
    except FileNotFoundError:
        return None


def add_retention_hold(
    *,
    runs_dir: Path,
    thread_id: str,
    request: RetentionHoldRequest,
) -> RetentionPolicy:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    policy = read_retention_policy_or_none(runs_dir, thread_id) or default_retention_policy(thread_id)
    hold_id = request.hold_id or f"hold-{len(policy.holds) + 1}"
    if any(hold.hold_id == hold_id and hold.active for hold in policy.holds):
        raise ValueError(f"Active hold already exists: {hold_id}")
    policy.holds.append(
        RetentionHold(
            hold_id=hold_id,
            reason=request.reason,
            requested_by=request.requested_by,
        )
    )
    policy.legal_hold = True
    policy.generated_at = now_iso_utc()
    write_retention_policy(run_dir, policy)
    return policy


def release_retention_hold(
    *,
    runs_dir: Path,
    thread_id: str,
    request: RetentionReleaseRequest,
) -> RetentionPolicy:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    policy = read_retention_policy(runs_dir, thread_id)
    active = policy.active_holds
    if request.hold_id:
        active = [hold for hold in active if hold.hold_id == request.hold_id]
    if not active:
        raise ValueError("No matching active retention hold found")
    now = now_iso_utc()
    for hold in active:
        hold.released_at = now
        hold.released_by = request.released_by
        hold.release_reason = request.reason
    policy.legal_hold = bool(policy.active_holds)
    policy.generated_at = now
    write_retention_policy(run_dir, policy)
    return policy


def default_retention_policy(thread_id: str) -> RetentionPolicy:
    now = datetime.now(timezone.utc)
    retain_until = now + timedelta(days=DEFAULT_RETENTION_DAYS["standard"])
    return RetentionPolicy(
        thread_id=thread_id,
        retention_class="standard",
        retain_until=_format_dt(retain_until),
        reason="Default retention policy.",
    )


def retention_blocks_cleanup(
    policy: RetentionPolicy | None,
    *,
    now: datetime | None = None,
) -> str | None:
    if policy is None:
        return None
    if policy.legal_hold or policy.active_holds:
        holds = ", ".join(hold.hold_id for hold in policy.active_holds) or "policy legal_hold"
        return f"retention hold active: {holds}"
    now = now or datetime.now(timezone.utc)
    retain_until = _parse_iso_datetime(policy.retain_until)
    if retain_until is not None and retain_until > now:
        return f"protected until {policy.retain_until}"
    return None


def write_retention_policy(run_dir: Path, policy: RetentionPolicy) -> list[str]:
    (run_dir / RETENTION_POLICY_JSON).write_text(
        json.dumps(_model_to_plain(policy), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / RETENTION_POLICY_MD).write_text(
        render_retention_policy_markdown(policy),
        encoding="utf-8",
    )
    return [RETENTION_POLICY_JSON, RETENTION_POLICY_MD]


def render_retention_policy_markdown(policy: RetentionPolicy) -> str:
    lines = [
        "# Retention Policy",
        "",
        f"- Thread ID: `{policy.thread_id}`",
        f"- Generated at: `{policy.generated_at}`",
        f"- Class: `{policy.retention_class}`",
        f"- Retain until: `{policy.retain_until or 'not set'}`",
        f"- Delete after: `{policy.delete_after or 'not set'}`",
        f"- Legal hold: `{policy.legal_hold}`",
        f"- Requested by: `{policy.requested_by}`",
        f"- Reason: {policy.reason or 'None'}",
        "",
        "## Holds",
        "",
    ]
    if policy.holds:
        for hold in policy.holds:
            status = "active" if hold.active else f"released at {hold.released_at}"
            lines.extend(
                [
                    f"### {hold.hold_id}",
                    "",
                    f"- Status: {status}",
                    f"- Created at: `{hold.created_at}`",
                    f"- Requested by: `{hold.requested_by}`",
                    f"- Reason: {hold.reason}",
                ]
            )
            if not hold.active:
                lines.extend(
                    [
                        f"- Released by: `{hold.released_by or 'unknown'}`",
                        f"- Release reason: {hold.release_reason or 'None'}",
                    ]
                )
            lines.append("")
    else:
        lines.append("- None")
    if policy.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in policy.warnings)
    return "\n".join(lines).rstrip() + "\n"


def _resolve_retain_until(
    now: datetime,
    request: RetentionPolicyRequest,
) -> datetime | None:
    if request.retain_until:
        return _parse_iso_datetime(request.retain_until)
    retain_days = (
        request.retain_days
        if request.retain_days is not None
        else DEFAULT_RETENTION_DAYS[request.retention_class]
    )
    if retain_days <= 0:
        return None
    return now + timedelta(days=retain_days)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except Exception as e:
        raise ValueError(f"Invalid ISO datetime: {value}") from e
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    run_dir = (root / thread_id).resolve()
    if root != run_dir and root not in run_dir.parents:
        raise ValueError("Invalid thread_id")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(thread_id)
    return run_dir


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
