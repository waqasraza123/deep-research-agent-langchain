import pytest
from pathlib import Path

from deep_research_agent.artifacts import (
    ensure_thread_dir,
    artifact_abs_path,
    safe_thread_id,
)


def test_safe_thread_id_rejects_bad_ids():
    with pytest.raises(ValueError):
        safe_thread_id("../evil")
    with pytest.raises(ValueError):
        safe_thread_id("a/b")
    with pytest.raises(ValueError):
        safe_thread_id("a\\b")


def test_artifact_abs_path_blocks_escape(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    tid = "ok"
    ensure_thread_dir(runs_dir, tid)

    with pytest.raises(ValueError):
        artifact_abs_path(runs_dir, tid, "../x")

    with pytest.raises(ValueError):
        artifact_abs_path(runs_dir, tid, "/etc/passwd")