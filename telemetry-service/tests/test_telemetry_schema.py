"""Contract validation for the telemetry ingest payload.

A payload that satisfies the contract reaches the use case and is refused with
501; a payload that violates it is rejected with 422 before it gets there. So
"accepted by the contract" is asserted as *not* 422.
"""

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TELEMETRY_URL, error_code

Mutate = Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]


def test_valid_payload_passes_contract_validation(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code != 422


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "event_time",
    [
        pytest.param("2026-09-01T04:30:00Z", id="utc"),
        pytest.param("2026-09-01T12:30:00+08:00", id="positive-offset"),
        pytest.param("2026-09-01T08:30:00-07:00", id="negative-offset"),
    ],
)
def test_timezone_aware_event_time_is_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, event_time: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", event_time))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


def test_naive_event_time_is_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", "2026-09-01T12:30:00"))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "NAIVE_TIMESTAMP"


def test_event_time_must_be_a_string_not_an_epoch_number(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_time", 1756700000))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


def test_source_offset_is_not_normalized_to_utc(valid_payload: dict[str, Any]) -> None:
    """Normalization to UTC is Phase 2; the contract layer must preserve the offset."""
    from app.api.schemas import TelemetryIngestRequest

    event = TelemetryIngestRequest.model_validate(valid_payload).to_domain_event()

    assert event.event_time.utcoffset().total_seconds() == 8 * 3600
    assert event.event_time.hour == 9


# --------------------------------------------------------------------------- #
# Units - validated, never converted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("unit", ["km/h", "mph", "m/s"])
def test_supported_speed_units_are_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, unit: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"]["speed"].__setitem__("unit", unit))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


@pytest.mark.parametrize("unit", ["knots", "KMH", "kph", ""])
def test_unsupported_speed_units_are_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, unit: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"]["speed"].__setitem__("unit", unit))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "UNSUPPORTED_UNIT"


@pytest.mark.parametrize("unit", ["degC", "degF", "K"])
def test_supported_temperature_units_are_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, unit: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"]["battery_temperature"].__setitem__("unit", unit))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


@pytest.mark.parametrize("unit", ["C", "celsius", "degc", "F"])
def test_unsupported_temperature_units_are_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, unit: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"]["battery_temperature"].__setitem__("unit", unit))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "UNSUPPORTED_UNIT"


def test_units_are_never_converted_in_this_phase(valid_payload: dict[str, Any]) -> None:
    """32.3 mph stays 32.3 mph. Conversion to km/h belongs to Phase 2."""
    from app.api.schemas import TelemetryIngestRequest
    from app.domain.units import MetricName

    event = TelemetryIngestRequest.model_validate(valid_payload).to_domain_event()

    assert event.metrics[MetricName.SPEED].value == 32.3
    assert event.metrics[MetricName.SPEED].unit == "mph"
    assert event.metrics[MetricName.BATTERY_TEMPERATURE].value == 96.4
    assert event.metrics[MetricName.BATTERY_TEMPERATURE].unit == "degF"


def test_unit_is_required_on_every_measurement(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    """There is no default unit: an omitted unit is an error, not an assumption."""
    payload = payload_with(valid_payload, lambda p: p["metrics"]["speed"].pop("unit"))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


def test_request_level_unit_system_is_not_part_of_the_contract(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("unit_system", "imperial"))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


# --------------------------------------------------------------------------- #
# Metric values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("unit", "value"),
    [("percent", -0.1), ("percent", 100.1), ("fraction", -0.1), ("fraction", 1.1)],
)
def test_soc_outside_its_units_range_is_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, unit: str, value: float
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"].__setitem__("soc", {"value": value, "unit": unit}))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


@pytest.mark.parametrize(("unit", "value"), [("percent", 0), ("percent", 100), ("fraction", 0.42)])
def test_soc_inside_its_units_range_is_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, unit: str, value: float
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"].__setitem__("soc", {"value": value, "unit": unit}))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


def test_negative_speed_is_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"]["speed"].__setitem__("value", -1.0))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


def test_negative_motor_rpm_is_accepted_because_no_convention_is_documented(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    """The contract documents no rotation-direction convention, so none is invented."""
    payload = payload_with(valid_payload, lambda p: p["metrics"]["motor_rpm"].__setitem__("value", -50))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


@pytest.mark.parametrize("value", ["32.3", "1,5", True, None, [], {}])
def test_non_numeric_metric_values_are_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, value: Any
) -> None:
    """Locale-independent numeric API: a number is a JSON number, never a string."""
    payload = payload_with(valid_payload, lambda p: p["metrics"]["speed"].__setitem__("value", value))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


def test_integer_metric_values_are_accepted(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"]["motor_rpm"].__setitem__("value", 4120))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


# --------------------------------------------------------------------------- #
# Metric set
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "metric",
    ["soc", "battery_voltage", "battery_current", "battery_temperature", "speed", "motor_rpm"],
)
def test_every_metric_is_required(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, metric: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p["metrics"].pop(metric))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "MISSING_METRIC"


def test_unknown_metrics_are_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(
        valid_payload, lambda p: p["metrics"].__setitem__("tyre_pressure", {"value": 2.4, "unit": "bar"})
    )

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "UNKNOWN_METRIC"


# --------------------------------------------------------------------------- #
# Identity and schema version
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["event_id", "vehicle_id", "site_id"])
def test_identifiers_are_required(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, field: str
) -> None:
    payload = payload_with(valid_payload, lambda p: p.pop(field))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


@pytest.mark.parametrize("field", ["event_id", "vehicle_id", "site_id"])
@pytest.mark.parametrize("value", ["", "   ", 42, None])
def test_identifiers_must_be_non_blank_strings(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate, field: str, value: Any
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__(field, value))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422


def test_event_id_is_opaque_and_need_not_be_a_uuid(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    """event_id is a domain identifier, not a storage key and not a typed value."""
    payload = payload_with(valid_payload, lambda p: p.__setitem__("event_id", "site-taipei-01:seq-000917"))

    assert client.post(TELEMETRY_URL, json=payload).status_code != 422


def test_unsupported_schema_version_is_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("schema_version", "2.0"))

    response = client.post(TELEMETRY_URL, json=payload)

    assert response.status_code == 422
    assert error_code(response) == "UNSUPPORTED_SCHEMA_VERSION"


def test_unknown_top_level_fields_are_rejected(
    client: TestClient, valid_payload: dict[str, Any], payload_with: Mutate
) -> None:
    payload = payload_with(valid_payload, lambda p: p.__setitem__("received_at", "2026-08-31T01:14:23Z"))

    assert client.post(TELEMETRY_URL, json=payload).status_code == 422
