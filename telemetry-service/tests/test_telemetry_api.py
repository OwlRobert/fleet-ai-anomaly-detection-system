"""Ingest endpoint behaviour: honest refusal, never a fabricated success."""

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import TELEMETRY_URL


def test_valid_ingest_is_stored(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 201
    assert response.json()["duplicate"] is False


def test_ingest_never_fabricates_an_inference_verdict(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    """The event is stored and unscored. No model ran, so nothing is claimed."""
    inference = client.post(TELEMETRY_URL, json=valid_payload).json()["inference"]

    assert inference == {
        "status": "PENDING",
        "is_anomaly": None,
        "anomaly_score": None,
        "model_name": None,
        "model_version": None,
        "error_code": None,
    }


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
