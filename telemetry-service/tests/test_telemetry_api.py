"""Ingest endpoint behaviour: honest refusal, never a fabricated success."""

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import TELEMETRY_URL, error_code


def test_valid_ingest_is_refused_rather_than_faked(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 501
    assert error_code(response) == "NOT_IMPLEMENTED"


def test_ingest_never_fabricates_persistence_or_inference(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    """No stored record, no anomaly verdict, no received_at, no duplicate flag."""
    body = client.post(TELEMETRY_URL, json=valid_payload).json()

    assert set(body) == {"error"}
    serialized = str(body)
    for fabricated in ("is_anomaly", "anomaly_score", "received_at", "duplicate", "COMPLETED"):
        assert fabricated not in serialized


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
