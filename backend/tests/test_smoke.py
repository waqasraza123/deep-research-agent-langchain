from deep_research_agent.api import app
from fastapi.testclient import TestClient


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True