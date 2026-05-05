from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.runs.cleanup import build_cleanup_plan
from deep_research_agent.runs.contracts import (
    ResumePointKind,
    ReviewState,
    RunCancellationRequest,
    RunStatus,
    utc_now,
)
from deep_research_agent.runs.repository import RunListFilters, RunRepository
from deep_research_agent.runs.resumability import inspect_resume_point
from deep_research_agent.runs.review import approve_review, request_changes
from deep_research_agent.runs.state_machine import InvalidRunTransitionError


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(tmp_path / "runs")


def _create(repo: RunRepository, thread_id: str = "run-1", *, require_review: bool = False):
    return repo.create(
        thread_id=thread_id,
        question="What is the test question?",
        urls=["https://example.com"],
        settings_snapshot={"model_provider": "mock"},
        require_review=require_review,
    )


def _advance_to_building(repo: RunRepository, thread_id: str) -> None:
    repo.transition(thread_id, RunStatus.PLANNING)
    repo.transition(thread_id, RunStatus.FETCHING_SOURCES)
    repo.transition(thread_id, RunStatus.ANALYZING)
    repo.transition(thread_id, RunStatus.WRITING_REPORT)
    repo.transition(thread_id, RunStatus.BUILDING_EVIDENCE)


def test_state_transitions_and_invalid_transition(tmp_path: Path):
    repo = _repo(tmp_path)
    _create(repo)

    repo.transition("run-1", RunStatus.PLANNING)
    repo.transition("run-1", RunStatus.FETCHING_SOURCES)

    with pytest.raises(InvalidRunTransitionError):
        repo.transition("run-1", RunStatus.COMPLETED)


def test_registry_persistence(tmp_path: Path):
    repo = _repo(tmp_path)
    _create(repo)
    repo.transition("run-1", RunStatus.PLANNING)
    repo.set_warnings("run-1", ["warning"])

    reloaded = RunRepository(tmp_path / "runs").get("run-1")

    assert reloaded.thread_id == "run-1"
    assert reloaded.status == RunStatus.PLANNING
    assert reloaded.warnings == ["warning"]


def test_resumability_detection_from_artifacts(tmp_path: Path):
    repo = _repo(tmp_path)
    _create(repo)
    td = ensure_thread_dir(repo.runs_dir, "run-1")
    (td / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (td / "sources.json").write_text("[]\n", encoding="utf-8")

    point = inspect_resume_point(repo.runs_dir, "run-1", repo.get("run-1"))

    assert point.resumable is True
    assert point.point == ResumePointKind.AFTER_SOURCE_FETCHING


def test_cancellation_marker(tmp_path: Path):
    repo = _repo(tmp_path)
    _create(repo)

    run = repo.request_cancellation(
        "run-1",
        RunCancellationRequest(requested_by="operator", reason="stop"),
    )

    assert run.cancellation is not None
    assert repo.cancellation_requested("run-1") is True


def test_review_workflow(tmp_path: Path):
    repo = _repo(tmp_path)
    _create(repo, require_review=True)
    _advance_to_building(repo, "run-1")
    repo.transition("run-1", RunStatus.WAITING_FOR_REVIEW)

    review = request_changes(
        repo,
        "run-1",
        reviewer="alice",
        notes="Needs tighter citations",
        requested_changes=["Add source quotes"],
    )
    assert review.status == ReviewState.CHANGES_REQUESTED
    assert review.requested_changes == ["Add source quotes"]

    approved = approve_review(repo, "run-1", reviewer="alice", notes="Looks good")
    assert approved.status == ReviewState.APPROVED
    assert repo.get("run-1").status == RunStatus.COMPLETED


def test_run_listing_filters(tmp_path: Path):
    repo = _repo(tmp_path)
    _create(repo, "completed")
    _advance_to_building(repo, "completed")
    repo.transition("completed", RunStatus.COMPLETED)

    _create(repo, "failed")
    repo.transition("failed", RunStatus.PLANNING)
    repo.record_error("failed", "boom", fail_run=True)

    failed = repo.list(RunListFilters(status=RunStatus.FAILED, has_errors=True))

    assert [run.thread_id for run in failed] == ["failed"]


def test_cleanup_planning(tmp_path: Path):
    repo = _repo(tmp_path)
    run = _create(repo)
    repo.transition("run-1", RunStatus.PLANNING)
    run = repo.get("run-1")
    run.updated_at = utc_now() - timedelta(hours=48)
    repo.save(run, touch=False)

    plan = build_cleanup_plan(repo, stale_after_hours=24, max_dir_bytes=1_000_000)

    assert [item.thread_id for item in plan.items] == ["run-1"]
    assert "stale incomplete run" in plan.items[0].reason


def test_review_api_and_run_filters(client):
    r = client.post(
        "/run",
        json={
            "question": "test question",
            "mock_mode": True,
            "require_review": True,
        },
    )
    assert r.status_code == 200
    tid = r.json()["thread_id"]
    assert r.json()["run"]["status"] == "waiting_for_review"

    review = client.post(
        f"/runs/{tid}/review/approve",
        json={"reviewer": "operator", "notes": "approved"},
    )
    assert review.status_code == 200
    assert review.json()["status"] == "approved"

    runs = client.get("/runs", params={"status": "completed", "review_status": "approved"})
    assert runs.status_code == 200
    assert any(item["thread_id"] == tid for item in runs.json())
