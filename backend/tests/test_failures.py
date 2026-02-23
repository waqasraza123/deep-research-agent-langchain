def test_run_failure_returns_500_and_backfills(failing_client):
    r = failing_client.post("/run", json={"question": "test question"})
    assert r.status_code == 500

    detail = r.json()["detail"]
    assert "thread_id" in detail
    tid = detail["thread_id"]

    # even on failure, artifacts endpoint should show required files
    lr = failing_client.get(f"/threads/{tid}/artifacts")
    assert lr.status_code == 200
    paths = {a["path"] for a in lr.json()}

    assert "plan.md" in paths
    assert "notes.md" in paths
    assert "sources.json" in paths
    assert "report.md" in paths