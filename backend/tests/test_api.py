import json


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_run_creates_required_artifacts(client):
    r = client.post("/run", json={"question": "test question", "urls": ["https://example.com"]})
    assert r.status_code == 200

    body = r.json()
    assert "thread_id" in body
    tid = body["thread_id"]

    paths = {a["path"] for a in body["artifacts"]}
    assert "run.json" in paths
    assert "plan.md" in paths
    assert "notes.md" in paths
    assert "sources.json" in paths
    assert "report.md" in paths
    assert "evidence_ledger.json" in paths
    assert "evidence_coverage.json" in paths

    # sources.json should be valid json
    sr = client.get(f"/threads/{tid}/artifacts/sources.json")
    assert sr.status_code == 200
    data = json.loads(sr.text)
    assert isinstance(data, list)


def test_artifacts_list_and_download(client):
    r = client.post("/run", json={"question": "test question"})
    tid = r.json()["thread_id"]

    lr = client.get(f"/runs/{tid}/artifacts")
    assert lr.status_code == 200
    items = lr.json()
    assert any(i["path"] == "report.md" for i in items)

    dr = client.get(f"/runs/{tid}/artifacts/report.md")
    assert dr.status_code == 200
    assert "test report" in dr.text.lower()

    legacy = client.get(f"/threads/{tid}/artifacts/report.md")
    assert legacy.status_code == 200


def test_artifact_path_traversal_blocked(client):
    r = client.post("/run", json={"question": "test question"})
    tid = r.json()["thread_id"]

    bad = client.get(f"/threads/{tid}/artifacts/../.env")
    assert bad.status_code in (400, 404)

    bad2 = client.get(f"/threads/{tid}/artifacts/%2e%2e/%2e%2e/.env")
    assert bad2.status_code in (400, 404)


def test_evidence_rebuild_route(client):
    r = client.post("/run", json={"question": "test question"})
    tid = r.json()["thread_id"]

    er = client.post(f"/runs/{tid}/evidence/rebuild")
    assert er.status_code == 200
    body = er.json()
    assert body["thread_id"] == tid
    assert "total_claims" in body["coverage"]
    assert any(a["path"] == "evidence_ledger.md" for a in body["artifacts"])


def test_research_plan_route_creates_strategy_artifacts(client):
    r = client.post(
        "/research-plan",
        json={
            "question": "Compare LangGraph and CrewAI for a production research agent backend",
            "thread_id": "planning-test",
        },
    )
    assert r.status_code == 200

    body = r.json()
    assert body["thread_id"] == "planning-test"
    assert body["strategy"]["intent"] == "comparative_analysis"

    paths = {a["path"] for a in body["artifacts"]}
    assert "strategy.json" in paths
    assert "strategy.md" in paths
    assert "subquestions.json" in paths
    assert "verification_plan.md" in paths


def test_mock_run_writes_intelligent_artifacts_and_public_run_snapshot(client):
    r = client.post(
        "/run",
        json={
            "question": "Validate integrated mock backend artifacts",
            "mock_mode": True,
            "urls": ["https://example.invalid/root"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    tid = body["thread_id"]
    paths = {a["path"] for a in body["artifacts"]}

    assert "run.json" in paths
    assert "source_graph.json" in paths
    assert "source_graph.md" in paths
    assert "evidence_ledger.json" in paths
    assert "evidence_coverage.json" in paths

    run_snapshot = client.get(f"/runs/{tid}/artifacts/run.json")
    assert run_snapshot.status_code == 200
    assert run_snapshot.json()["thread_id"] == tid
