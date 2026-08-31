"""Health endpoint."""

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "telemetry-service", "version": "0.1.0"}


def test_health_reports_nothing_about_dependencies(client: TestClient) -> None:
    """Phase 1 has no database and no inference, so health must claim neither."""
    body = client.get("/health").json()

    assert set(body) == {"status", "service", "version"}
