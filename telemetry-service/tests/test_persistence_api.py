"""Ingestion and history through the HTTP boundary, backed by the store."""

from datetime import timedelta
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FIXED_NOW, TELEMETRY_URL, error_code
from tests.factories import scored, stored_event
from tests.fakes import InMemoryTelemetryRepository, UnavailableTelemetryRepository

Mutate = Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]

HISTORY_URL = "/api/v1/vehicles/veh-tw-0142/telemetry"
ANOMALIES_URL = "/api/v1/vehicles/veh-tw-0142/anomalies"
RANGE = {
    "start": (FIXED_NOW - timedelta(days=7)).isoformat(),
    "end": (FIXED_NOW + timedelta(days=1)).isoformat(),
}


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def test_a_first_event_is_created(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 201
    assert response.json()["duplicate"] is False
    assert response.json()["inference"]["status"] == "PENDING"


def test_an_exact_retry_returns_200_and_the_original(
    client: TestClient, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository
) -> None:
    first = client.post(TELEMETRY_URL, json=valid_payload).json()

    retry = client.post(TELEMETRY_URL, json=valid_payload)

    assert retry.status_code == 200
    assert retry.json()["duplicate"] is True
    assert retry.json()["received_at"] == first["received_at"]
    assert len(repository.events) == 1


def test_a_conflicting_reuse_keeps_the_stored_event(
    client: TestClient,
    valid_payload: dict[str, Any],
    payload_with: Mutate,
    repository: InMemoryTelemetryRepository,
) -> None:
    """First write wins: the original is returned, never overwritten."""
    client.post(TELEMETRY_URL, json=valid_payload)
    conflicting = payload_with(valid_payload, lambda p: p.__setitem__("vehicle_id", "veh-cz-0007"))

    response = client.post(TELEMETRY_URL, json=conflicting)

    assert response.status_code == 200
    assert response.json()["vehicle_id"] == "veh-tw-0142"
    assert len(repository.events) == 1
    assert repository.events[0].event.vehicle_id == "veh-tw-0142"


def test_the_response_carries_no_storage_identity(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    body = client.post(TELEMETRY_URL, json=valid_payload).json()

    assert set(body) == {
        "event_id",
        "vehicle_id",
        "site_id",
        "event_time",
        "received_at",
        "duplicate",
        "metrics",
        "inference",
    }


def test_an_unstorable_event_is_not_acknowledged(
    app, valid_payload: dict[str, Any]
) -> None:
    """Fail-closed: 503, so the client's idempotent retry is the recovery path."""
    from app.api.dependencies import get_telemetry_repository

    app.dependency_overrides[get_telemetry_repository] = UnavailableTelemetryRepository

    response = TestClient(app).post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 503
    assert error_code(response) == "PERSISTENCE_UNAVAILABLE"


def test_a_persistence_failure_reveals_nothing_about_the_database(
    app, valid_payload: dict[str, Any]
) -> None:
    from app.api.dependencies import get_telemetry_repository

    app.dependency_overrides[get_telemetry_repository] = UnavailableTelemetryRepository

    response = TestClient(app).post(TELEMETRY_URL, json=valid_payload)

    for leak in ("mongo", "Mongo", "27017", "pymongo", "Traceback"):
        assert leak not in response.text


# --------------------------------------------------------------------------- #
# History, end to end
# --------------------------------------------------------------------------- #


def test_an_ingested_event_appears_in_the_history(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    client.post(TELEMETRY_URL, json=valid_payload)

    page = client.get(HISTORY_URL, params=RANGE).json()

    assert page["count"] == 1
    assert page["items"][0]["event_id"] == valid_payload["event_id"]
    assert page["items"][0]["metrics"]["speed"] == pytest.approx(51.9818112)


def test_an_ingested_event_is_not_an_anomaly(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    """It was never scored, and unscored is not non-anomalous either way."""
    client.post(TELEMETRY_URL, json=valid_payload)

    assert client.get(ANOMALIES_URL, params=RANGE).json()["count"] == 0


def _all_keys(value: Any) -> set[str]:
    """Every JSON object key anywhere in a response body."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_history_items_carry_no_storage_identity(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    client.post(TELEMETRY_URL, json=valid_payload)

    response = client.get(HISTORY_URL, params=RANGE)

    assert "_id" not in _all_keys(response.json())
    assert "ObjectId" not in response.text


def test_a_query_failure_is_reported_as_unavailable(app) -> None:
    from app.api.dependencies import get_telemetry_repository

    app.dependency_overrides[get_telemetry_repository] = UnavailableTelemetryRepository

    response = TestClient(app).get(HISTORY_URL, params=RANGE)

    assert response.status_code == 503
    assert error_code(response) == "PERSISTENCE_UNAVAILABLE"


def test_anomalies_return_stored_scored_events(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    """Storage fixtures describing future scored documents. No model ran."""
    repository.events.extend(
        [
            stored_event(event_id="anomalous", inference=scored(is_anomaly=True)),
            stored_event(event_id="normal", inference=scored(is_anomaly=False)),
            stored_event(event_id="unscored"),
        ]
    )

    page = client.get(ANOMALIES_URL, params=RANGE).json()

    assert [item["event_id"] for item in page["items"]] == ["anomalous"]
    assert page["items"][0]["inference"]["status"] == "COMPLETED"
