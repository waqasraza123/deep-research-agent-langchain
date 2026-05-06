from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .contracts import (
    CancellationRequest,
    ControlRequestStatus,
    DeadLetterRecord,
    LeaseStatus,
    PauseRequest,
    ResearchJob,
    ResearchJobStatus,
    ResearchStage,
    ResearchStageRecord,
    RuntimeErrorRecord,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeLease,
    StageStatus,
    utc_now,
)
from .errors import RuntimeLeaseError, RuntimeNotFoundError
from .idempotency import redact_settings_snapshot
from .state_machine import validate_job_transition, validate_stage_transition

ACTIVE_IDEMPOTENT_STATUSES = {
    ResearchJobStatus.QUEUED,
    ResearchJobStatus.LEASED,
    ResearchJobStatus.RUNNING,
    ResearchJobStatus.PAUSING,
    ResearchJobStatus.PAUSED,
    ResearchJobStatus.RESUME_REQUESTED,
    ResearchJobStatus.CANCELLING,
    ResearchJobStatus.FAILED,
}


def _dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _json(value: Any) -> str:
    if hasattr(value, "dict"):
        value = value.dict()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class RuntimeRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def from_settings(cls, settings: Any) -> "RuntimeRepository":
        raw = getattr(settings, "runtime_sqlite_path", None)
        path = Path(raw) if raw else Path(settings.runs_dir) / "runtime.sqlite3"
        return cls(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    idempotency_key TEXT,
                    question TEXT NOT NULL,
                    urls_json TEXT NOT NULL,
                    settings_snapshot_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    failed_at TEXT,
                    cancelled_at TEXT,
                    paused_at TEXT,
                    resume_requested_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    retry_after TEXT,
                    budget_json TEXT NOT NULL,
                    budget_usage_json TEXT NOT NULL,
                    error_json TEXT,
                    warnings_json TEXT NOT NULL,
                    artifact_summary_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority);
                CREATE INDEX IF NOT EXISTS idx_jobs_thread_id ON jobs(thread_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs(idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_jobs_retry_after ON jobs(retry_after);

                CREATE TABLE IF NOT EXISTS stages (
                    stage_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    required INTEGER NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    failed_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    input_artifacts_json TEXT NOT NULL,
                    output_artifacts_json TEXT NOT NULL,
                    checkpoint_marker TEXT,
                    resumable INTEGER NOT NULL,
                    skip_reason TEXT,
                    error_json TEXT,
                    warnings_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE INDEX IF NOT EXISTS idx_stages_job_stage ON stages(job_id, stage);
                CREATE INDEX IF NOT EXISTS idx_stages_status ON stages(status);

                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    stage TEXT,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    artifact_refs_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_job_sequence ON events(job_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_events_thread_sequence
                ON events(thread_id, sequence);

                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    heartbeat_at TEXT,
                    heartbeat_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    lost_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_leases_job_status ON leases(job_id, status);
                CREATE INDEX IF NOT EXISTS idx_leases_expires ON leases(expires_at, status);

                CREATE TABLE IF NOT EXISTS control_requests (
                    request_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_control_job_kind ON control_requests(job_id, kind);

                CREATE TABLE IF NOT EXISTS queue (
                    job_id TEXT PRIMARY KEY,
                    priority INTEGER NOT NULL DEFAULT 0,
                    enqueued_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_queue_order ON queue(priority DESC, enqueued_at ASC);

                CREATE TABLE IF NOT EXISTS dead_letters (
                    job_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    moved_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    error_json TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS runtime_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def create_job(self, job: ResearchJob) -> ResearchJob:
        job.settings_snapshot = redact_settings_snapshot(job.settings_snapshot)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, thread_id, idempotency_key, question, urls_json,
                    settings_snapshot_json, status, stage, priority, created_at, updated_at,
                    started_at, completed_at, failed_at, cancelled_at, paused_at,
                    resume_requested_at, attempts, max_attempts, retry_after, budget_json,
                    budget_usage_json, error_json, warnings_json, artifact_summary_json,
                    metadata_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                self._job_params(job),
            )
        return job

    def _job_params(self, job: ResearchJob) -> tuple[Any, ...]:
        return (
            job.job_id,
            job.thread_id,
            job.idempotency_key,
            job.question,
            _json(job.urls),
            _json(job.settings_snapshot),
            job.status,
            job.stage,
            job.priority,
            _dt(job.created_at),
            _dt(job.updated_at),
            _dt(job.started_at),
            _dt(job.completed_at),
            _dt(job.failed_at),
            _dt(job.cancelled_at),
            _dt(job.paused_at),
            _dt(job.resume_requested_at),
            job.attempts,
            job.max_attempts,
            _dt(job.retry_after),
            _json(job.budget),
            _json(job.budget_usage),
            _json(job.error) if job.error else None,
            _json(job.warnings),
            _json(job.artifact_summary),
            _json(job.metadata),
        )

    def _row_to_job(self, row: sqlite3.Row) -> ResearchJob:
        data = dict(row)
        error = _loads(data.get("error_json"), None)
        return ResearchJob(
            job_id=data["job_id"],
            thread_id=data["thread_id"],
            idempotency_key=data["idempotency_key"],
            question=data["question"],
            urls=_loads(data["urls_json"], []),
            settings_snapshot=_loads(data["settings_snapshot_json"], {}),
            status=ResearchJobStatus(data["status"]),
            stage=ResearchStage(data["stage"]),
            priority=int(data["priority"]),
            created_at=_parse_dt(data["created_at"]) or utc_now(),
            updated_at=_parse_dt(data["updated_at"]) or utc_now(),
            started_at=_parse_dt(data["started_at"]),
            completed_at=_parse_dt(data["completed_at"]),
            failed_at=_parse_dt(data["failed_at"]),
            cancelled_at=_parse_dt(data["cancelled_at"]),
            paused_at=_parse_dt(data["paused_at"]),
            resume_requested_at=_parse_dt(data["resume_requested_at"]),
            attempts=int(data["attempts"]),
            max_attempts=int(data["max_attempts"]),
            retry_after=_parse_dt(data["retry_after"]),
            budget=_loads(data["budget_json"], {}),
            budget_usage=_loads(data["budget_usage_json"], {}),
            error=RuntimeErrorRecord(**error) if isinstance(error, dict) else None,
            warnings=_loads(data["warnings_json"], []),
            artifact_summary=_loads(data["artifact_summary_json"], {}),
            metadata=_loads(data["metadata_json"], {}),
            lease=self.get_active_lease(data["job_id"]),
        )

    def get_job(self, job_id: str) -> ResearchJob:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise RuntimeNotFoundError(f"Job not found: {job_id}")
        return self._row_to_job(row)

    def get_job_by_thread_id(self, thread_id: str) -> ResearchJob | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE thread_id = ?", (thread_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def get_job_by_idempotency_key(
        self, key: str, *, active_only: bool = True
    ) -> ResearchJob | None:
        sql = "SELECT * FROM jobs WHERE idempotency_key = ?"
        params: list[Any] = [key]
        if active_only:
            statuses = [status.value for status in ACTIVE_IDEMPOTENT_STATUSES]
            sql += f" AND status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        sql += " ORDER BY created_at DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._row_to_job(row) if row else None

    def update_job(self, job: ResearchJob) -> ResearchJob:
        job.updated_at = utc_now()
        job.settings_snapshot = redact_settings_snapshot(job.settings_snapshot)
        with self.transaction() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET
                    thread_id=?, idempotency_key=?, question=?, urls_json=?,
                    settings_snapshot_json=?, status=?, stage=?, priority=?, created_at=?,
                    updated_at=?, started_at=?, completed_at=?, failed_at=?, cancelled_at=?,
                    paused_at=?, resume_requested_at=?, attempts=?, max_attempts=?,
                    retry_after=?, budget_json=?, budget_usage_json=?, error_json=?,
                    warnings_json=?, artifact_summary_json=?, metadata_json=?
                WHERE job_id=?
                """,
                (*self._job_params(job)[1:], job.job_id),
            )
            if result.rowcount == 0:
                raise RuntimeNotFoundError(f"Job not found: {job.job_id}")
        return job

    def list_jobs(
        self,
        *,
        status: ResearchJobStatus | None = None,
        stage: ResearchStage | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        has_errors: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ResearchJob]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if stage:
            clauses.append("stage = ?")
            params.append(stage.value)
        if created_after:
            clauses.append("created_at >= ?")
            params.append(_dt(created_after))
        if created_before:
            clauses.append("created_at <= ?")
            params.append(_dt(created_before))
        if has_errors is not None:
            clauses.append("error_json IS NOT NULL" if has_errors else "error_json IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.extend([max(1, min(limit, 500)), max(0, offset)])
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def mark_job_status(
        self,
        job_id: str,
        status: ResearchJobStatus,
        *,
        stage: ResearchStage | None = None,
        validate: bool = True,
        explicit_restore: bool = False,
    ) -> ResearchJob:
        job = self.get_job(job_id)
        if validate:
            validate_job_transition(job.status, status, explicit_restore=explicit_restore)
        now = utc_now()
        job.status = status
        job.updated_at = now
        if stage is not None:
            job.stage = stage
        if status == ResearchJobStatus.RUNNING and job.started_at is None:
            job.started_at = now
        elif status == ResearchJobStatus.COMPLETED:
            job.completed_at = now
            job.stage = ResearchStage.COMPLETED
        elif status == ResearchJobStatus.FAILED:
            job.failed_at = now
            job.stage = ResearchStage.FAILED
        elif status == ResearchJobStatus.CANCELLED:
            job.cancelled_at = now
            job.stage = ResearchStage.CANCELLED
        elif status == ResearchJobStatus.PAUSED:
            job.paused_at = now
        elif status == ResearchJobStatus.RESUME_REQUESTED:
            job.resume_requested_at = now
        elif status == ResearchJobStatus.DEAD_LETTERED:
            job.stage = ResearchStage.FAILED
        return self.update_job(job)

    def append_warning(self, job_id: str, warning: str) -> ResearchJob:
        job = self.get_job(job_id)
        if warning not in job.warnings:
            job.warnings.append(warning)
        return self.update_job(job)

    def set_error(self, job_id: str, error: RuntimeErrorRecord) -> ResearchJob:
        job = self.get_job(job_id)
        job.error = error
        return self.update_job(job)

    def mark_completed(self, job_id: str) -> ResearchJob:
        return self.mark_job_status(job_id, ResearchJobStatus.COMPLETED, validate=False)

    def mark_failed(self, job_id: str, error: RuntimeErrorRecord | None = None) -> ResearchJob:
        job = self.get_job(job_id)
        if error:
            job.error = error
        self.update_job(job)
        return self.mark_job_status(job_id, ResearchJobStatus.FAILED, validate=False)

    def mark_dead_lettered(self, job_id: str, reason: str) -> ResearchJob:
        self.move_to_dead_letter(job_id, reason=reason)
        return self.mark_job_status(job_id, ResearchJobStatus.DEAD_LETTERED, validate=False)

    def create_stage_record(self, record: ResearchStageRecord) -> ResearchStageRecord:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO stages (
                    stage_id, job_id, thread_id, stage, status, required, started_at,
                    completed_at, failed_at, attempts, input_artifacts_json,
                    output_artifacts_json, checkpoint_marker, resumable, skip_reason,
                    error_json, warnings_json, metrics_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.stage_id,
                    record.job_id,
                    record.thread_id,
                    record.stage,
                    record.status,
                    int(record.required),
                    _dt(record.started_at),
                    _dt(record.completed_at),
                    _dt(record.failed_at),
                    record.attempts,
                    _json(record.input_artifacts),
                    _json(record.output_artifacts),
                    record.checkpoint_marker,
                    int(record.resumable),
                    record.skip_reason,
                    _json(record.error) if record.error else None,
                    _json(record.warnings),
                    _json(record.metrics),
                    _dt(now),
                    _dt(now),
                ),
            )
        return record

    def _row_to_stage(self, row: sqlite3.Row) -> ResearchStageRecord:
        err = _loads(row["error_json"], None)
        return ResearchStageRecord(
            stage_id=row["stage_id"],
            job_id=row["job_id"],
            thread_id=row["thread_id"],
            stage=ResearchStage(row["stage"]),
            status=StageStatus(row["status"]),
            required=bool(row["required"]),
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
            failed_at=_parse_dt(row["failed_at"]),
            attempts=int(row["attempts"]),
            input_artifacts=_loads(row["input_artifacts_json"], []),
            output_artifacts=_loads(row["output_artifacts_json"], []),
            checkpoint_marker=row["checkpoint_marker"],
            resumable=bool(row["resumable"]),
            skip_reason=row["skip_reason"],
            error=RuntimeErrorRecord(**err) if isinstance(err, dict) else None,
            warnings=_loads(row["warnings_json"], []),
            metrics=_loads(row["metrics_json"], {}),
        )

    def update_stage_record(self, record: ResearchStageRecord) -> ResearchStageRecord:
        with self.transaction() as conn:
            result = conn.execute(
                """
                UPDATE stages SET status=?, started_at=?, completed_at=?, failed_at=?,
                    attempts=?, input_artifacts_json=?, output_artifacts_json=?,
                    checkpoint_marker=?, resumable=?, skip_reason=?, error_json=?,
                    warnings_json=?, metrics_json=?, updated_at=?
                WHERE stage_id=?
                """,
                (
                    record.status,
                    _dt(record.started_at),
                    _dt(record.completed_at),
                    _dt(record.failed_at),
                    record.attempts,
                    _json(record.input_artifacts),
                    _json(record.output_artifacts),
                    record.checkpoint_marker,
                    int(record.resumable),
                    record.skip_reason,
                    _json(record.error) if record.error else None,
                    _json(record.warnings),
                    _json(record.metrics),
                    _dt(utc_now()),
                    record.stage_id,
                ),
            )
            if result.rowcount == 0:
                raise RuntimeNotFoundError(f"Stage not found: {record.stage_id}")
        return record

    def get_stage_records(self, job_id: str) -> list[ResearchStageRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stages WHERE job_id=? ORDER BY created_at ASC",
                (job_id,),
            ).fetchall()
        return [self._row_to_stage(row) for row in rows]

    def get_latest_stage(self, job_id: str) -> ResearchStageRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM stages WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return self._row_to_stage(row) if row else None

    def _stage_for_update(self, job_id: str, stage: ResearchStage) -> ResearchStageRecord:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM stages WHERE job_id=? AND stage=? ORDER BY created_at DESC LIMIT 1",
                (job_id, stage.value),
            ).fetchone()
        if row:
            return self._row_to_stage(row)
        job = self.get_job(job_id)
        record = ResearchStageRecord(
            stage_id=str(uuid.uuid4()),
            job_id=job.job_id,
            thread_id=job.thread_id,
            stage=stage,
        )
        return self.create_stage_record(record)

    def mark_stage_started(self, job_id: str, stage: ResearchStage) -> ResearchStageRecord:
        record = self._stage_for_update(job_id, stage)
        validate_stage_transition(record.status, StageStatus.RUNNING)
        record.status = StageStatus.RUNNING
        record.started_at = utc_now()
        record.attempts += 1
        return self.update_stage_record(record)

    def mark_stage_completed(
        self,
        stage_id: str,
        *,
        output_artifacts: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ResearchStageRecord:
        record = self._get_stage_by_id(stage_id)
        validate_stage_transition(record.status, StageStatus.COMPLETED)
        record.status = StageStatus.COMPLETED
        record.completed_at = utc_now()
        if output_artifacts is not None:
            record.output_artifacts = output_artifacts
        if metrics:
            record.metrics.update(metrics)
        return self.update_stage_record(record)

    def mark_stage_failed(
        self,
        stage_id: str,
        error: RuntimeErrorRecord,
    ) -> ResearchStageRecord:
        record = self._get_stage_by_id(stage_id)
        validate_stage_transition(record.status, StageStatus.FAILED)
        record.status = StageStatus.FAILED
        record.failed_at = utc_now()
        record.error = error
        return self.update_stage_record(record)

    def mark_stage_skipped(
        self,
        job_id: str,
        stage: ResearchStage,
        reason: str,
    ) -> ResearchStageRecord:
        record = self._stage_for_update(job_id, stage)
        validate_stage_transition(record.status, StageStatus.SKIPPED)
        record.status = StageStatus.SKIPPED
        record.skip_reason = reason
        record.completed_at = utc_now()
        return self.update_stage_record(record)

    def _get_stage_by_id(self, stage_id: str) -> ResearchStageRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM stages WHERE stage_id=?", (stage_id,)).fetchone()
        if row is None:
            raise RuntimeNotFoundError(f"Stage not found: {stage_id}")
        return self._row_to_stage(row)

    def append_event(self, event: RuntimeEvent) -> RuntimeEvent:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, job_id, thread_id, timestamp, event_type, stage,
                    severity, message, data_json, artifact_refs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.job_id,
                    event.thread_id,
                    _dt(event.timestamp),
                    event.event_type,
                    str(event.stage) if event.stage else None,
                    event.severity,
                    event.message,
                    _json(event.data),
                    _json(event.artifact_refs),
                ),
            )
        return event

    def _row_to_event(self, row: sqlite3.Row) -> RuntimeEvent:
        return RuntimeEvent(
            event_id=row["event_id"],
            job_id=row["job_id"],
            thread_id=row["thread_id"],
            timestamp=_parse_dt(row["timestamp"]) or utc_now(),
            event_type=RuntimeEventType(row["event_type"]),
            stage=ResearchStage(row["stage"]) if row["stage"] else None,
            severity=row["severity"],
            message=row["message"],
            data=_loads(row["data_json"], {}),
            artifact_refs=_loads(row["artifact_refs_json"], []),
        )

    def list_events(self, job_id: str, *, since_event_id: str | None = None) -> list[RuntimeEvent]:
        params: list[Any] = [job_id]
        where = "WHERE job_id=?"
        if since_event_id:
            where += " AND sequence > COALESCE((SELECT sequence FROM events WHERE event_id=?), -1)"
            params.append(since_event_id)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY sequence ASC",
                params,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_events_by_thread(
        self, thread_id: str, *, since_event_id: str | None = None
    ) -> list[RuntimeEvent]:
        params: list[Any] = [thread_id]
        where = "WHERE thread_id=?"
        if since_event_id:
            where += " AND sequence > COALESCE((SELECT sequence FROM events WHERE event_id=?), -1)"
            params.append(since_event_id)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY sequence ASC",
                params,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def count_events(self, job_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE job_id=?", (job_id,)
            ).fetchone()
            return int(row[0])

    def acquire_lease(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> RuntimeLease:
        now = utc_now()
        expires = now + timedelta(seconds=lease_seconds)
        lease = RuntimeLease(
            lease_id=str(uuid.uuid4()),
            job_id=job_id,
            worker_id=worker_id,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires,
        )
        with self.transaction() as conn:
            active = conn.execute(
                "SELECT lease_id FROM leases WHERE job_id=? AND status=? AND expires_at > ?",
                (job_id, LeaseStatus.ACTIVE.value, _dt(now)),
            ).fetchone()
            if active:
                raise RuntimeLeaseError(f"Job already leased: {job_id}")
            conn.execute(
                "UPDATE leases SET status=?, lost_reason=? WHERE job_id=? AND status=?",
                (LeaseStatus.EXPIRED.value, "superseded", job_id, LeaseStatus.ACTIVE.value),
            )
            conn.execute(
                """
                INSERT INTO leases (
                    lease_id, job_id, worker_id, acquired_at, expires_at, heartbeat_at,
                    heartbeat_count, status, lost_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lease.lease_id,
                    lease.job_id,
                    lease.worker_id,
                    _dt(lease.acquired_at),
                    _dt(lease.expires_at),
                    _dt(lease.heartbeat_at),
                    lease.heartbeat_count,
                    str(lease.status),
                    lease.lost_reason,
                ),
            )
        return lease

    def _row_to_lease(self, row: sqlite3.Row) -> RuntimeLease:
        return RuntimeLease(
            lease_id=row["lease_id"],
            job_id=row["job_id"],
            worker_id=row["worker_id"],
            acquired_at=_parse_dt(row["acquired_at"]) or utc_now(),
            expires_at=_parse_dt(row["expires_at"]) or utc_now(),
            heartbeat_at=_parse_dt(row["heartbeat_at"]),
            heartbeat_count=int(row["heartbeat_count"]),
            status=LeaseStatus(row["status"]),
            lost_reason=row["lost_reason"],
        )

    def heartbeat_lease(self, lease_id: str, *, lease_seconds: int) -> RuntimeLease:
        now = utc_now()
        expires = now + timedelta(seconds=lease_seconds)
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None:
                raise RuntimeLeaseError(f"Lease not found: {lease_id}")
            lease = self._row_to_lease(row)
            if lease.status != LeaseStatus.ACTIVE:
                raise RuntimeLeaseError(f"Lease is not active: {lease_id}")
            conn.execute(
                """
                UPDATE leases SET heartbeat_at=?, heartbeat_count=heartbeat_count+1,
                    expires_at=? WHERE lease_id=?
                """,
                (_dt(now), _dt(expires), lease_id),
            )
        return self._get_lease(lease_id)

    def release_lease(self, lease_id: str, *, status: LeaseStatus = LeaseStatus.RELEASED) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE leases SET status=? WHERE lease_id=? AND status=?",
                (str(status), lease_id, LeaseStatus.ACTIVE.value),
            )

    def expire_stale_leases(self) -> list[RuntimeLease]:
        now = utc_now()
        expired: list[RuntimeLease] = []
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM leases WHERE status=? AND expires_at <= ?",
                (LeaseStatus.ACTIVE.value, _dt(now)),
            ).fetchall()
            expired = [self._row_to_lease(row) for row in rows]
            conn.execute(
                "UPDATE leases SET status=?, lost_reason=? WHERE status=? AND expires_at <= ?",
                (
                    LeaseStatus.EXPIRED.value,
                    "lease expired without heartbeat",
                    LeaseStatus.ACTIVE.value,
                    _dt(now),
                ),
            )
        return expired

    def get_active_lease(self, job_id: str) -> RuntimeLease | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM leases
                WHERE job_id=? AND status=?
                ORDER BY acquired_at DESC LIMIT 1
                """,
                (job_id, LeaseStatus.ACTIVE.value),
            ).fetchone()
        return self._row_to_lease(row) if row else None

    def list_expired_leases(self) -> list[RuntimeLease]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM leases WHERE status=? ORDER BY expires_at DESC",
                (LeaseStatus.EXPIRED.value,),
            ).fetchall()
        return [self._row_to_lease(row) for row in rows]

    def _get_lease(self, lease_id: str) -> RuntimeLease:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        if row is None:
            raise RuntimeLeaseError(f"Lease not found: {lease_id}")
        return self._row_to_lease(row)

    def request_cancel(self, request: CancellationRequest) -> CancellationRequest:
        return self._write_control("cancel", request)

    def get_cancellation_request(self, job_id: str) -> CancellationRequest | None:
        data = self._read_control(job_id, "cancel")
        return CancellationRequest(**data) if data else None

    def request_pause(self, request: PauseRequest) -> PauseRequest:
        return self._write_control("pause", request)

    def get_pause_request(self, job_id: str) -> PauseRequest | None:
        data = self._read_control(job_id, "pause")
        return PauseRequest(**data) if data else None

    def request_resume(self, request: Any) -> Any:
        return self._write_control("resume", request)

    def get_resume_request(self, job_id: str) -> Any | None:
        data = self._read_control(job_id, "resume")
        if not data:
            return None
        from .contracts import ResumeRequest

        return ResumeRequest(**data)

    def acknowledge_control_request(self, job_id: str, kind: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE control_requests SET status=?, updated_at=?
                WHERE job_id=? AND kind=? AND status=?
                """,
                (
                    ControlRequestStatus.ACKNOWLEDGED.value,
                    _dt(utc_now()),
                    job_id,
                    kind,
                    ControlRequestStatus.REQUESTED.value,
                ),
            )

    def _write_control(self, kind: str, request: Any) -> Any:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO control_requests (
                    request_id, job_id, kind, payload_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{request.job_id}:{kind}",
                    request.job_id,
                    kind,
                    _json(request),
                    str(request.status),
                    _dt(now),
                    _dt(now),
                ),
            )
        return request

    def _read_control(self, job_id: str, kind: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM control_requests
                WHERE job_id=? AND kind=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (job_id, kind),
            ).fetchone()
        return _loads(row["payload_json"], None) if row else None

    def enqueue_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO queue(job_id, priority, enqueued_at) VALUES (?, ?, ?)",
                (job_id, job.priority, _dt(utc_now())),
            )

    def dequeue_next_job(self) -> ResearchJob | None:
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT jobs.* FROM queue
                JOIN jobs ON jobs.job_id = queue.job_id
                WHERE jobs.status = ? AND (jobs.retry_after IS NULL OR jobs.retry_after <= ?)
                ORDER BY queue.priority DESC, queue.enqueued_at ASC
                LIMIT 1
                """,
                (ResearchJobStatus.QUEUED.value, _dt(now)),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM queue WHERE job_id=?", (row["job_id"],))
        return self._row_to_job(row)

    def requeue_job(self, job_id: str, *, retry_after: datetime | None = None) -> ResearchJob:
        job = self.get_job(job_id)
        job.retry_after = retry_after
        job.status = ResearchJobStatus.QUEUED
        self.update_job(job)
        self.enqueue_job(job_id)
        return job

    def list_queue(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM queue ORDER BY priority DESC, enqueued_at ASC"
            ).fetchall()
        return [row["job_id"] for row in rows]

    def move_to_dead_letter(self, job_id: str, *, reason: str) -> DeadLetterRecord:
        job = self.get_job(job_id)
        record = DeadLetterRecord(
            job_id=job.job_id,
            thread_id=job.thread_id,
            reason=reason,
            error=job.error,
            attempts=job.attempts,
        )
        with self.transaction() as conn:
            conn.execute("DELETE FROM queue WHERE job_id=?", (job_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO dead_letters
                (job_id, thread_id, moved_at, reason, error_json, attempts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.thread_id,
                    _dt(record.moved_at),
                    record.reason,
                    _json(record.error) if record.error else None,
                    record.attempts,
                ),
            )
        return record

    def list_dead_letters(self) -> list[DeadLetterRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM dead_letters ORDER BY moved_at DESC").fetchall()
        records: list[DeadLetterRecord] = []
        for row in rows:
            err = _loads(row["error_json"], None)
            records.append(
                DeadLetterRecord(
                    job_id=row["job_id"],
                    thread_id=row["thread_id"],
                    moved_at=_parse_dt(row["moved_at"]) or utc_now(),
                    reason=row["reason"],
                    error=RuntimeErrorRecord(**err) if isinstance(err, dict) else None,
                    attempts=int(row["attempts"]),
                )
            )
        return records

    def restore_dead_letter(self, job_id: str) -> ResearchJob:
        with self.transaction() as conn:
            conn.execute("DELETE FROM dead_letters WHERE job_id=?", (job_id,))
        job = self.mark_job_status(
            job_id,
            ResearchJobStatus.QUEUED,
            validate=False,
            explicit_restore=True,
        )
        job.error = None
        job.retry_after = None
        self.update_job(job)
        self.enqueue_job(job_id)
        return job

    def status_counts(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}
