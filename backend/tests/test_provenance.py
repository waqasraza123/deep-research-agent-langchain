from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.provenance import (
    build_artifact_manifest,
    build_dependency_dag,
    build_replay_plan,
    build_reproducibility_report,
    diff_run_dirs,
    file_sha256,
    redact_secrets,
    refresh_provenance_artifacts,
    stable_hash,
)
from deep_research_agent.provenance.lineage import build_run_input_fingerprint, safe_run_dir
from deep_research_agent.runs.repository import RunRepository


def _create_run(tmp_path: Path, thread_id: str = "run-1"):
    runs_dir = tmp_path / "runs"
    repo = RunRepository(runs_dir)
    run = repo.create(
        thread_id=thread_id,
        question="What changed in the test system?",
        urls=["https://example.com/a/"],
        settings_snapshot={
            "model_provider": "openai",
            "openai_model": "gpt-test",
            "openai_api_key": "sk-test-secret",
            "temperature": 0.2,
        },
    )
    td = ensure_thread_dir(runs_dir, thread_id)
    (td / "sources").mkdir()
    (td / "sources" / "s1.txt").write_text("Captured source text.\n", encoding="utf-8")
    (td / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "S1",
                    "url": "https://example.com/a/",
                    "normalized_url": "https://example.com/a",
                    "title": "Example",
                    "local_path": "sources/s1.txt",
                    "fetched_at": "2026-05-05T00:00:00Z",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (td / "context_packs.json").write_text("{}\n", encoding="utf-8")
    (td / "report.md").write_text("# Report\n\nGrounded in S1.\n", encoding="utf-8")
    (td / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "model_call_started",
                "message": "write report",
                "metadata": {"provider": "openai", "model_name": "gpt-test"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return repo, run, td


def test_content_hashing_is_deterministic(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("same\n", encoding="utf-8")

    assert file_sha256(path) == file_sha256(path)
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_input_fingerprint_is_deterministic(tmp_path: Path):
    _repo, run, _td = _create_run(tmp_path)

    first = build_run_input_fingerprint(run)
    second = build_run_input_fingerprint(run)

    assert first == second
    assert first.normalized_urls == ["https://example.com/a"]


def test_manifest_generation_and_secret_redaction(tmp_path: Path):
    _repo, run, td = _create_run(tmp_path)

    manifest = build_artifact_manifest(td, "run-1", run)

    report = next(item for item in manifest.artifacts if item.artifact_path == "report.md")
    assert report.producer_subsystem == "agent"
    assert report.source_dependencies[0].identifier == "S1"
    assert manifest.sources[0].content_hash == file_sha256(td / "sources" / "s1.txt")
    assert redact_secrets({"openai_api_key": "sk-test-secret"})["openai_api_key"] == "[REDACTED]"
    assert "sk-test-secret" not in json.dumps(
        [m.redacted_config for m in manifest.model_invocations]
    )


def test_dependency_dag_generation(tmp_path: Path):
    _repo, run, td = _create_run(tmp_path)
    manifest = build_artifact_manifest(td, "run-1", run)

    graph = build_dependency_dag(manifest)

    assert any(node["id"] == "artifact:report.md" for node in graph.nodes)
    assert any(edge["target"] == "artifact:report.md" for edge in graph.edges)


def test_reproducibility_report_and_replay_plan(tmp_path: Path):
    _repo, run, td = _create_run(tmp_path)
    manifest = build_artifact_manifest(td, "run-1", run)

    report = build_reproducibility_report(manifest, run)
    plan = build_replay_plan(manifest, report)

    assert report.status == "partially_replayable"
    assert "OPENAI_API_KEY" in report.credentials_required
    assert any(step["step_id"] == "compare" for step in plan.ordered_steps)
    assert plan.expected_output_hashes["report.md"]


def test_refresh_writes_provenance_artifacts(tmp_path: Path):
    repo, run, td = _create_run(tmp_path)

    refresh_provenance_artifacts(repo.runs_dir, run.thread_id, run=run)

    assert (td / "artifact_manifest.json").exists()
    assert (td / "artifact_dependency_dag.json").exists()
    assert (td / "reproducibility_report.json").exists()
    assert (td / "replay_plan.json").exists()


def test_artifact_diffing(tmp_path: Path):
    repo, run, _td = _create_run(tmp_path, "left")
    _repo2, right_run, right_td = _create_run(tmp_path, "right")
    (right_td / "report.md").write_text("# Report\n\nChanged.\n", encoding="utf-8")
    (right_td / "extra.md").write_text("extra\n", encoding="utf-8")

    summary = diff_run_dirs(
        repo.runs_dir,
        "left",
        "right",
        left_run=run,
        right_run=right_run,
    )

    assert "extra.md" in summary.added_artifacts
    assert "report.md" in summary.changed_artifacts
    assert "report.md" in summary.changed_hashes


def test_provenance_api_endpoints(client, test_runs_dir: Path):
    tid = "prov-api"
    td = ensure_thread_dir(test_runs_dir, tid)
    (td / "sources.json").write_text("[]\n", encoding="utf-8")
    (td / "report.md").write_text("# Report\n", encoding="utf-8")

    assert client.get(f"/runs/{tid}/manifest").status_code == 200
    assert client.get(f"/runs/{tid}/provenance").status_code == 200
    assert client.get(f"/runs/{tid}/reproducibility").status_code == 200
    replay = client.get(f"/runs/{tid}/replay-plan")
    assert replay.status_code == 200
    assert replay.json()["thread_id"] == tid

    other = ensure_thread_dir(test_runs_dir, "prov-api-2")
    (other / "sources.json").write_text("[]\n", encoding="utf-8")
    diff = client.post(
        "/runs/diff",
        json={"left_thread_id": tid, "right_thread_id": "prov-api-2"},
    )
    assert diff.status_code == 200


def test_provenance_path_safety(tmp_path: Path):
    with pytest.raises(ValueError):
        safe_run_dir(tmp_path / "runs", "../bad")
