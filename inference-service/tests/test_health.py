"""Health endpoint, with and without a model."""

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "inference-service",
        "version": "0.1.0",
        "model_loaded": True,
    }


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    assert client.get("/health").json()["model_loaded"] is True


def test_health_never_claims_a_model_that_failed_to_load(unloaded_client: TestClient) -> None:
    """The one claim this endpoint must never get wrong."""
    body = unloaded_client.get("/health").json()

    assert body["model_loaded"] is False
    assert body["status"] == "ok"


def test_liveness_survives_a_failed_artifact_load(unloaded_client: TestClient) -> None:
    """The process is alive; it just cannot score. That is not a 5xx."""
    assert unloaded_client.get("/health").status_code == 200


def test_health_reveals_nothing_about_the_filesystem(unloaded_client: TestClient) -> None:
    body = unloaded_client.get("/health").text

    for leak in ("/", "joblib", "Traceback", "artifact"):
        assert leak not in body
