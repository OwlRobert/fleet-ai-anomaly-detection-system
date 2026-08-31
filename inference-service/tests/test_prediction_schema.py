"""The canonical feature contract, and everything it refuses."""

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from tests.conftest import PREDICT_URL

Mutate = Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]

FEATURES = ["soc", "battery_voltage", "battery_current", "battery_temperature", "speed", "motor_rpm"]


def test_canonical_request_passes_contract_validation(
    client: TestClient, canonical_request: dict[str, Any]
) -> None:
    assert client.post(PREDICT_URL, json=canonical_request).status_code != 422


# --------------------------------------------------------------------------- #
# Source units must not leak across the boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("feature", FEATURES)
def test_source_measurement_objects_are_rejected(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate, feature: str
) -> None:
    """`{"value": 32.3, "unit": "mph"}` belongs to the telemetry contract, not this one."""
    body = request_with(
        canonical_request, lambda b: b["features"].__setitem__(feature, {"value": 32.3, "unit": "mph"})
    )

    assert client.post(PREDICT_URL, json=body).status_code == 422


@pytest.mark.parametrize(
    ("key", "value"),
    [("unit", "mph"), ("units", {"speed": "mph"}), ("source_units", {"battery_temperature": "degF"})],
)
def test_unit_bearing_keys_are_rejected_inside_features(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate, key: str, value: Any
) -> None:
    body = request_with(canonical_request, lambda b: b["features"].__setitem__(key, value))

    assert client.post(PREDICT_URL, json=body).status_code == 422


@pytest.mark.parametrize("key", ["unit_system", "source_units", "vehicle_id", "event_time"])
def test_unit_bearing_and_telemetry_keys_are_rejected_at_the_top_level(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate, key: str
) -> None:
    body = request_with(canonical_request, lambda b: b.__setitem__(key, "mph"))

    assert client.post(PREDICT_URL, json=body).status_code == 422


def test_no_source_unit_string_can_reach_the_model(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate
) -> None:
    """A payload shaped like source telemetry is refused outright."""
    body = request_with(
        canonical_request,
        lambda b: b["features"].update(
            speed={"value": 32.3, "unit": "mph"},
            battery_temperature={"value": 96.4, "unit": "degF"},
        ),
    )

    response = client.post(PREDICT_URL, json=body)

    assert response.status_code == 422
    assert "mph" not in response.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# Feature values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("feature", FEATURES)
def test_every_feature_is_required(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate, feature: str
) -> None:
    body = request_with(canonical_request, lambda b: b["features"].pop(feature))

    assert client.post(PREDICT_URL, json=body).status_code == 422


def test_unknown_features_are_rejected(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate
) -> None:
    body = request_with(canonical_request, lambda b: b["features"].__setitem__("tyre_pressure", 2.4))

    assert client.post(PREDICT_URL, json=body).status_code == 422


@pytest.mark.parametrize("value", ["51.98", "1,5", True, None, [], {}])
def test_non_numeric_feature_values_are_rejected(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate, value: Any
) -> None:
    """Locale-independent numeric API: `"1,5"` is a string, not a number."""
    body = request_with(canonical_request, lambda b: b["features"].__setitem__("speed", value))

    assert client.post(PREDICT_URL, json=body).status_code == 422


def test_integer_feature_values_are_accepted(
    client: TestClient, canonical_request: dict[str, Any], request_with: Mutate
) -> None:
    body = request_with(canonical_request, lambda b: b["features"].__setitem__("motor_rpm", 4120))

    assert client.post(PREDICT_URL, json=body).status_code != 422


def test_features_object_is_required(client: TestClient) -> None:
    assert client.post(PREDICT_URL, json={}).status_code == 422
