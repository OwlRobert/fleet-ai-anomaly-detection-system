"""Health endpoint."""

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "inference-service", "version": "0.1.0"}


def test_health_makes_no_claim_about_a_model(client: TestClient) -> None:
    """There is no model loading yet, so `model_loaded` must be absent, not false-but-meaningless."""
    body = client.get("/health").json()

    assert set(body) == {"status", "service", "version"}
    assert "model_loaded" not in body
