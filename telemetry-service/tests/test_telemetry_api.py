"""Ingest endpoint behaviour: honest refusal, never a fabricated success."""

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import TELEMETRY_URL
from tests.fakes import StubInferencePort


def test_valid_ingest_is_stored(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 201
    assert response.json()["duplicate"] is False


def test_ingest_returns_the_models_actual_verdict(
    client: TestClient, valid_payload: dict[str, Any], inference: StubInferencePort
) -> None:
    """Whatever the model said is what the client is told. Nothing invented."""
    body = client.post(TELEMETRY_URL, json=valid_payload).json()["inference"]

    assert body == {
        "status": "COMPLETED",
        "is_anomaly": False,
        "anomaly_score": -0.1034,
        "model_name": "isolation-forest-telemetry",
        "model_version": "0.1.0",
        "error_code": None,
    }
    assert inference.call_count == 1


def test_error_responses_use_the_contract_envelope(client: TestClient) -> None:
    body = client.post(TELEMETRY_URL, json={}).json()

    assert set(body["error"]) == {"code", "message", "details"}
    assert isinstance(body["error"]["code"], str)


def test_validation_errors_are_machine_readable_per_field(client: TestClient) -> None:
    body = client.post(TELEMETRY_URL, json={"schema_version": "1.0"}).json()

    errors = body["error"]["details"]["errors"]
    assert {"code", "field", "message"} == set(errors[0])
    assert {"body.event_id", "body.vehicle_id", "body.site_id"} <= {error["field"] for error in errors}


def test_error_responses_never_leak_a_stack_trace(client: TestClient, valid_payload: dict[str, Any]) -> None:
    for response in (
        client.post(TELEMETRY_URL, json=valid_payload),
        client.post(TELEMETRY_URL, json={}),
    ):
        assert "Traceback" not in response.text
        assert "app/" not in response.text
