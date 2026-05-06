from __future__ import annotations

import json
from dataclasses import replace

from fastapi.testclient import TestClient

from deep_research_agent.api import create_app
from deep_research_agent.runtime_control.contracts import ResearchJobStatus
from deep_research_agent.runtime_control.queue import RuntimeQueue
from deep_research_agent.runtime_control.repository import RuntimeRepository
from deep_research_agent.runtime_control.worker import RuntimeWorker


def test_worker_processes_mock_job_and_writes_runtime_artifacts(test_settings, test_runs_dir):
    settings = replace(test_settings, runtime_mock_agent_execution_enabled=True)
    repo = RuntimeRepository(test_runs_dir / "runtime.sqlite3")
    queue = RuntimeQueue(repository=repo, runs_dir=test_runs_dir)
    job, created = queue.submit_job(
        question="What should runtime tests verify?",
        urls=["https://example.com/a"],
        settings_snapshot={"openai_api_key": "secret"},
        metadata={"mock_agent_execution": True},
    )

    assert created is True
    final = RuntimeWorker(settings=settings, repository=repo).process_next_job()

    assert final.status == ResearchJobStatus.COMPLETED
    td = test_runs_dir / job.thread_id
    for rel in (
        "runtime_input_snapshot.json",
        "runtime_job.json",
        "runtime_stages.json",
        "runtime_events.jsonl",
        "runtime_events.md",
        "runtime_budget.json",
        "runtime_final_summary.json",
        "plan.md",
        "notes.md",
        "sources.json",
        "report.md",
    ):
        assert (td / rel).exists(), rel
    assert "MOCK OUTPUT" in (td / "report.md").read_text(encoding="utf-8")
    events = [json.loads(line) for line in (td / "runtime_events.jsonl").read_text().splitlines()]
    assert "job_completed" in {event["event_type"] for event in events}


def test_runtime_api_submit_run_now_events_budget_and_artifacts(test_settings, test_runs_dir):
    settings = replace(test_settings, runtime_mock_agent_execution_enabled=True)
    client = TestClient(create_app(settings=settings))

    submitted = client.post(
        "/runtime/jobs",
        json={
            "question": "What should runtime API tests verify?",
            "urls": ["https://example.com/a"],
            "run_now": True,
            "mock_agent_execution": True,
        },
    )
    assert submitted.status_code == 200
    body = submitted.json()
    assert body["status"] == "completed"
    job_id = body["job_id"]
    thread_id = body["thread_id"]

    assert client.get(f"/runtime/jobs/{job_id}").json()["thread_id"] == thread_id
    assert client.get(f"/runtime/jobs/{job_id}/stages").json()
    assert client.get(f"/runtime/jobs/{job_id}/events").json()
    assert "usage" in client.get(f"/runtime/jobs/{job_id}/budget").json()
    assert client.get(f"/runs/{thread_id}/runtime").json()["job_id"] == job_id
    assert client.get(f"/runs/{thread_id}/events").json()
    assert client.get(f"/runs/{thread_id}/budget").json()["usage"]

    artifacts = client.get(f"/runs/{thread_id}/artifacts").json()
    paths = {item["path"] for item in artifacts}
    assert "runtime_final_summary.json" in paths
    assert client.get(f"/runs/{thread_id}/artifacts/runtime_job.json").status_code == 200
    assert client.get(f"/runs/{thread_id}/artifacts/%2E%2E/runtime_job.json").status_code == 400


def test_runtime_async_run_mode_preserves_sync_default(test_settings):
    client = TestClient(create_app(settings=test_settings))
    queued = client.post(
        "/run",
        json={
            "question": "What should async compatibility submit?",
            "urls": ["https://example.com/a"],
            "runtime_async": True,
            "mock_mode": True,
        },
    )
    assert queued.status_code == 200
    assert queued.json()["runtime_mode"] == "async_runtime"
    assert queued.json()["status"] == "queued"


def test_pause_resume_and_cancel_queued_job(test_settings):
    client = TestClient(create_app(settings=test_settings))
    submitted = client.post(
        "/runtime/jobs",
        json={
            "question": "What should queued control do?",
            "urls": [],
            "mock_agent_execution": True,
        },
    ).json()
    job_id = submitted["job_id"]

    paused = client.post(f"/runtime/jobs/{job_id}/pause", json={"reason": "hold"})
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    resumed = client.post(f"/runtime/jobs/{job_id}/resume", json={"reason": "continue"})
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "queued"
    cancelled = client.post(f"/runtime/jobs/{job_id}/cancel", json={"reason": "stop"})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
