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
    assert "plan.md" in paths
    assert "notes.md" in paths
    assert "sources.json" in paths
    assert "report.md" in paths

    # sources.json should be valid json
    sr = client.get(f"/threads/{tid}/artifacts/sources.json")
    assert sr.status_code == 200
    data = json.loads(sr.text)
    assert isinstance(data, list)


def test_artifacts_list_and_download(client):
    r = client.post("/run", json={"question": "test question"})
    tid = r.json()["thread_id"]

    lr = client.get(f"/threads/{tid}/artifacts")
    assert lr.status_code == 200
    items = lr.json()
    assert any(i["path"] == "report.md" for i in items)

    dr = client.get(f"/threads/{tid}/artifacts/report.md")
    assert dr.status_code == 200
    assert "test report" in dr.text.lower()


def test_artifact_path_traversal_blocked(client):
    r = client.post("/run", json={"question": "test question"})
    tid = r.json()["thread_id"]

    bad = client.get(f"/threads/{tid}/artifacts/../.env")
    assert bad.status_code in (400, 404)

    bad2 = client.get(f"/threads/{tid}/artifacts/%2e%2e/%2e%2e/.env")
    assert bad2.status_code in (400, 404)