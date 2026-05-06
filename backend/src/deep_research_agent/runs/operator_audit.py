"""Append-only operator action audit artifacts with per-file hash-chain checks."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import now_iso_utc, safe_thread_id
from deep_research_agent.provenance.lineage import redact_secrets

GLOBAL_AUDIT_DIR = "_audit"
OPERATOR_AUDIT_JSONL = "operator_audit.jsonl"
OPERATOR_AUDIT_MD = "operator_audit.md"


class OperatorAuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"op-{uuid.uuid4().hex[:12]}")
    created_at: str = Field(default_factory=now_iso_utc)
    event_type: str
    actor: str = "operator"
    thread_id: str | None = None
    affected_thread_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str | None = None
    event_hash: str = ""


class OperatorAuditVerification(BaseModel):
    scope: str
    event_count: int = 0
    valid: bool = True
    broken_at_event_id: str | None = None
    expected_previous_hash: str | None = None
    actual_previous_hash: str | None = None
    last_hash: str | None = None
    warnings: list[str] = Field(default_factory=list)


def record_operator_event(
    *,
    runs_dir: Path,
    event_type: str,
    actor: str = "operator",
    summary: str = "",
    thread_id: str | None = None,
    affected_thread_ids: list[str] | None = None,
    artifacts: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> OperatorAuditEvent:
    actor = actor.strip() if actor and actor.strip() else "operator"
    affected = _safe_thread_ids(affected_thread_ids or ([thread_id] if thread_id else []))
    event = OperatorAuditEvent(
        event_type=event_type,
        actor=actor,
        thread_id=thread_id,
        affected_thread_ids=affected,
        summary=summary,
        artifacts=sorted(set(artifacts or [])),
        metadata=redact_secrets(metadata or {}),
    )
    global_event = _append_event(_global_audit_path(runs_dir), event)
    for affected_thread_id in affected:
        run_dir = _run_dir_or_none(runs_dir, affected_thread_id)
        if run_dir is None:
            continue
        _append_event(run_dir / OPERATOR_AUDIT_JSONL, event)
        _write_markdown(
            run_dir / OPERATOR_AUDIT_MD,
            read_operator_events(runs_dir, affected_thread_id),
        )
    _write_markdown(_global_audit_markdown_path(runs_dir), read_operator_events(runs_dir))
    return global_event


def read_operator_events(
    runs_dir: Path,
    thread_id: str | None = None,
    *,
    limit: int = 500,
) -> list[OperatorAuditEvent]:
    path = _global_audit_path(runs_dir) if thread_id is None else _run_audit_path(runs_dir, thread_id)
    events = _read_events(path)
    if limit > 0:
        return events[-limit:]
    return events


def verify_operator_audit(
    runs_dir: Path,
    thread_id: str | None = None,
) -> OperatorAuditVerification:
    path = _global_audit_path(runs_dir) if thread_id is None else _run_audit_path(runs_dir, thread_id)
    scope = "global" if thread_id is None else f"run:{thread_id}"
    events, parse_warnings = _read_event_records(path)
    if parse_warnings:
        return OperatorAuditVerification(
            scope=scope,
            event_count=len(events),
            valid=False,
            warnings=parse_warnings,
        )
    previous: str | None = None
    for event in events:
        if event.previous_hash != previous:
            return OperatorAuditVerification(
                scope=scope,
                event_count=len(events),
                valid=False,
                broken_at_event_id=event.event_id,
                expected_previous_hash=previous,
                actual_previous_hash=event.previous_hash,
                last_hash=previous,
            )
        expected_hash = _event_hash(event, previous)
        if expected_hash != event.event_hash:
            return OperatorAuditVerification(
                scope=scope,
                event_count=len(events),
                valid=False,
                broken_at_event_id=event.event_id,
                expected_previous_hash=expected_hash,
                actual_previous_hash=event.event_hash,
                last_hash=previous,
                warnings=["Event hash does not match event payload."],
            )
        previous = event.event_hash
    return OperatorAuditVerification(
        scope=scope,
        event_count=len(events),
        valid=True,
        last_hash=previous,
    )


def render_operator_audit_markdown(events: list[OperatorAuditEvent]) -> str:
    lines = [
        "# Operator Audit Trail",
        "",
        f"- Events: {len(events)}",
        "",
    ]
    for event in events:
        affected = ", ".join(event.affected_thread_ids) or "none"
        artifacts = ", ".join(f"`{item}`" for item in event.artifacts) or "none"
        lines.extend(
            [
                f"## {event.created_at} - {event.event_type}",
                "",
                f"- Event ID: `{event.event_id}`",
                f"- Actor: `{event.actor}`",
                f"- Thread: `{event.thread_id or 'global'}`",
                f"- Affected runs: {affected}",
                f"- Summary: {event.summary or 'None'}",
                f"- Artifacts: {artifacts}",
                f"- Previous hash: `{event.previous_hash or 'genesis'}`",
                f"- Event hash: `{event.event_hash}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _append_event(path: Path, event: OperatorAuditEvent) -> OperatorAuditEvent:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = _last_hash(path)
    payload = _model_to_plain(event)
    payload["previous_hash"] = previous_hash
    payload["event_hash"] = _event_hash_from_payload(payload, previous_hash)
    stored = _validate_event(payload)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_model_to_plain(stored), ensure_ascii=False, sort_keys=True) + "\n")
    return stored


def _read_events(path: Path) -> list[OperatorAuditEvent]:
    events, _ = _read_event_records(path)
    return events


def _read_event_records(path: Path) -> tuple[list[OperatorAuditEvent], list[str]]:
    if not path.exists() or path.is_dir():
        return [], []
    events: list[OperatorAuditEvent] = []
    warnings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(_validate_event(json.loads(line)))
        except Exception as e:
            warnings.append(
                f"Line {line_number} is not a valid operator audit event: "
                f"{type(e).__name__}: {e}"
            )
            continue
    return events, warnings


def _last_hash(path: Path) -> str | None:
    events, warnings = _read_event_records(path)
    if warnings:
        raise ValueError(
            "Cannot append to malformed operator audit log: " + "; ".join(warnings[:3])
        )
    return events[-1].event_hash if events else None


def _event_hash(event: OperatorAuditEvent, previous_hash: str | None) -> str:
    payload = _model_to_plain(event)
    payload["previous_hash"] = previous_hash
    payload["event_hash"] = ""
    return _event_hash_from_payload(payload, previous_hash)


def _event_hash_from_payload(payload: dict[str, Any], previous_hash: str | None) -> str:
    canonical = dict(payload)
    canonical["previous_hash"] = previous_hash
    canonical["event_hash"] = ""
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_markdown(path: Path, events: list[OperatorAuditEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_operator_audit_markdown(events), encoding="utf-8")


def _global_audit_path(runs_dir: Path) -> Path:
    return runs_dir / GLOBAL_AUDIT_DIR / OPERATOR_AUDIT_JSONL


def _global_audit_markdown_path(runs_dir: Path) -> Path:
    return runs_dir / GLOBAL_AUDIT_DIR / OPERATOR_AUDIT_MD


def _run_audit_path(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    return runs_dir / thread_id / OPERATOR_AUDIT_JSONL


def _run_dir_or_none(runs_dir: Path, thread_id: str) -> Path | None:
    safe_thread_id(thread_id)
    run_dir = (runs_dir / thread_id).resolve()
    root = runs_dir.resolve()
    if root != run_dir and root not in run_dir.parents:
        return None
    return run_dir if run_dir.exists() and run_dir.is_dir() else None


def _safe_thread_ids(thread_ids: list[str]) -> list[str]:
    out: list[str] = []
    for thread_id in thread_ids:
        if not thread_id:
            continue
        out.append(safe_thread_id(thread_id))
    return list(dict.fromkeys(out))


def _validate_event(payload: dict[str, Any]) -> OperatorAuditEvent:
    validate = getattr(OperatorAuditEvent, "model_validate", None)
    if callable(validate):
        return validate(payload)
    return OperatorAuditEvent.parse_obj(payload)


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
