"""Normalization as seen through the HTTP boundary.

These prove the endpoint really normalizes without pretending persistence
exists, and that the clock-skew bounds reach the wire as contract error codes.
The `client` fixture pins the server clock; `live_client` uses the real wiring.
"""

from datetime import timedelta
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FIXED_NOW, TELEMETRY_URL, error_code

Mutate = Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]


def at(offset: timedelta) -> str:
    return (FIXED_NOW + offset).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# A normalizable event still refuses to fake persistence
# --------------------------------------------------------------------------- #


def test_normalized_event_still_reports_the_pipeline_as_incomplete(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 501
    assert error_code(response) == "NOT_IMPLEMENTED"


def test_normalization_success_leaks_no_canonical_internals_to_the_client(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    """The canonical event is internal until persistence returns it."""
    body = client.post(TELEMETRY_URL, json=valid_payload).json()

    assert set(body) == {"error"}
    for internal in ("metrics", "source_units", "received_at", "canonical", "51.98"):
        assert internal not in str(body)


# --------------------------------------------------------------------------- #
# Clock-skew bounds surface as contract error codes
# --------------------------------------------------------------------------- #


def test_event_too_far_in_the_future_is_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(minutes=6))))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "CLOCK_SKEW_FUTURE"


def test_event_exactly_at_the_future_skew_limit_is_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(seconds=300))))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 501


def test_event_older_than_the_ingestion_window_is_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(days=-31))))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "EVENT_TOO_OLD"


def test_event_exactly_at_the_max_age_is_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(days=-30))))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 501


def test_a_delayed_event_is_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(hours=-9))))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 501


def test_skew_rejection_reports_both_timestamps_without_correcting_them(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    event_time = at(timedelta(hours=3))
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", event_time))

    details = client.post(TELEMETRY_URL, json=payload).json()["error"]["details"]

    assert details["event_time"] == (FIXED_NOW + timedelta(hours=3)).isoformat()
    assert details["received_at"] == FIXED_NOW.isoformat()
    assert details["limit_seconds"] == 300.0


def test_skew_rejection_never_leaks_a_stack_trace(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(days=400))))

    response = client.post(TELEMETRY_URL, json=payload)

    assert "Traceback" not in response.text
    assert "app/" not in response.text


# --------------------------------------------------------------------------- #
# Ordering is not a normalization concern
# --------------------------------------------------------------------------- #


def test_an_older_event_after_a_newer_one_is_still_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    """Processing order is not event-time order; both must normalize."""
    newer = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(minutes=-1))))
    older = payload_with(valid_payload, lambda p: p.__setitem__("event_time", at(timedelta(hours=-5))))

    assert client.post(TELEMETRY_URL, json=newer).status_code == 501
    assert client.post(TELEMETRY_URL, json=older).status_code == 501


# --------------------------------------------------------------------------- #
# The real wiring, with no dependency override
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("event_time", "expected_code"),
    [
        pytest.param("2999-01-01T00:00:00Z", "CLOCK_SKEW_FUTURE", id="far-future"),
        pytest.param("2000-01-01T00:00:00Z", "EVENT_TOO_OLD", id="long-past"),
    ],
)
def test_the_application_wires_the_real_server_clock_and_configured_bounds(
    live_client: TestClient,
    valid_payload: dict[str, Any],
    payload_with: Mutate,
    event_time: str,
    expected_code: str,
) -> None:
    """Absurd timestamps give the same verdict whenever the suite is run."""
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", event_time))

    response = live_client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == expected_code
