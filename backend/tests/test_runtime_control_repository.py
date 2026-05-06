from __future__ import annotations

from datetime import timedelta

from deep_research_agent.runtime_control.contracts import (
    ResearchJob,
    ResearchJobStatus,
    ResearchStage,
    ResearchStageRecord,
    RuntimeEvent,
    RuntimeEventType,
    utc_now,
)
from deep_research_agent.runtime_control.idempotency import redact_settings_snapshot
from deep_research_agent.runtime_control.repository import RuntimeRepository


def test_repository_create_get_update_idempotency_and_redaction(tmp_path):
    repo = RuntimeRepository(tmp_path / "runtime.sqlite3")
    job = ResearchJob(
        job_id="job-1",
        thread_id="thread-1",
        idempotency_key="idem",
        question="What happened?",
        urls=["https://example.com"],
        settings_snapshot={"openai_api_key": "secret", "model": "mock"},
    )
    repo.create_job(job)

    loaded = repo.get_job("job-1")
    assert loaded.thread_id == "thread-1"
    assert loaded.settings_snapshot["openai_api_key"] == "[REDACTED]"
    assert repo.get_job_by_idempotency_key("idem").job_id == "job-1"

    repo.append_warning("job-1", "careful")
    assert repo.get_job("job-1").warnings == ["careful"]

    redacted = redact_settings_snapshot({"nested": {"token": "abc"}, "safe": "yes"})
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["safe"] == "yes"


def test_repository_stage_event_lease_queue_and_dead_letter(tmp_path):
    repo = RuntimeRepository(tmp_path / "runtime.sqlite3")
    job = repo.create_job(
        ResearchJob(job_id="job-1", thread_id="thread-1", question="What happened?")
    )
    repo.enqueue_job(job.job_id)
    assert repo.dequeue_next_job().job_id == job.job_id

    record = repo.create_stage_record(
        ResearchStageRecord(
            stage_id="stage-1",
            job_id=job.job_id,
            thread_id=job.thread_id,
            stage=ResearchStage.PLANNING,
        )
    )
    repo.mark_stage_started(job.job_id, ResearchStage.PLANNING)
    repo.mark_stage_completed(record.stage_id, output_artifacts=["runtime_plan.md"])
    assert repo.get_latest_stage(job.job_id).status == "completed"

    repo.append_event(
        RuntimeEvent(
            event_id="event-1",
            job_id=job.job_id,
            thread_id=job.thread_id,
            event_type=RuntimeEventType.JOB_QUEUED,
        )
    )
    assert repo.list_events(job.job_id)[0].event_id == "event-1"

    lease = repo.acquire_lease(job.job_id, worker_id="w1", lease_seconds=30)
    lease = repo.heartbeat_lease(lease.lease_id, lease_seconds=30)
    assert lease.heartbeat_count == 1
    repo.release_lease(lease.lease_id)
    assert repo.get_active_lease(job.job_id) is None

    expired = repo.acquire_lease(job.job_id, worker_id="w1", lease_seconds=1)
    with repo.transaction() as conn:
        conn.execute(
            "UPDATE leases SET expires_at=? WHERE lease_id=?",
            ((utc_now() - timedelta(seconds=5)).isoformat(), expired.lease_id),
        )
    assert repo.expire_stale_leases()[0].lease_id == expired.lease_id

    repo.mark_dead_lettered(job.job_id, "failed")
    assert repo.list_dead_letters()[0].job_id == job.job_id
    restored = repo.restore_dead_letter(job.job_id)
    assert restored.status == ResearchJobStatus.QUEUED

