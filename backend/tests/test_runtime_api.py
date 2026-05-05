from __future__ import annotations


def test_diagnostics_endpoint(client):
    r = client.get("/runtime/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["active_provider"] == "ollama"
    assert "models" in body
    assert "default_budget" in body


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    providers = {item["provider"] for item in r.json()}
    assert "mock" in providers
    assert "ollama" in providers


def test_mock_mode_run_requires_no_credentials(client):
    r = client.post(
        "/run",
        json={
            "question": "test question for mock mode",
            "mock_mode": True,
            "urls": ["https://example.invalid/no-network"],
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["mock"] is True
    assert "MOCK OUTPUT" in body["summary"]

    tid = body["thread_id"]
    report = client.get(f"/threads/{tid}/artifacts/report.md")
    assert report.status_code == 200
    assert "MOCK OUTPUT" in report.text

    artifacts = client.get(f"/runs/{tid}/artifacts")
    assert artifacts.status_code == 200
    paths = {item["path"] for item in artifacts.json()}
    assert "source_graph.json" in paths
    assert "evidence_ledger.json" in paths

    events = client.get(f"/runs/{tid}/events")
    assert events.status_code == 200
    event_types = [event["event_type"] for event in events.json()]
    assert "run_started" in event_types
    assert "run_completed" in event_types

    budget = client.get(f"/runs/{tid}/budget")
    assert budget.status_code == 200
    assert "usage" in budget.json()


def test_budget_exceeded_returns_429(client):
    r = client.post(
        "/run",
        json={
            "question": "test question with tiny model budget",
            "budget": {
                "max_model_calls": 0,
                "max_source_fetches": 3,
                "max_generated_chars": 80000,
                "max_runtime_seconds": 180,
                "max_artifacts_size": 5000000,
                "max_crawl_expansion": 10,
            },
        },
    )

    assert r.status_code == 429
    assert "model_calls" in r.json()["detail"]["reasons"][0]
