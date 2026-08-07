"""FastAPI guardrail integration tests (TestClient, no full model load)."""

from fastapi.testclient import TestClient

from serving import api as api_module
from serving.api import app

client = TestClient(app)


def test_health_includes_guardrail_config():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "guardrails" in body
    assert body["guardrails"]["latency_sla_ms"] == 500


def test_recommend_503_without_artifacts():
    api_module.candidate_agent = None
    resp = client.post("/recommend", json={"user_id": "U00001", "top_k": 5})
    assert resp.status_code == 503


def test_search_validation_422():
    resp = client.post("/search", json={"top_k": 5})
    assert resp.status_code == 422


def test_complete_the_look_validation_negative_seed():
    resp = client.post("/complete-the-look", json={"seed_item_idx": -1, "top_k": 5})
    assert resp.status_code == 422
