"""Focused tests for the simulator's generation and reporting.

The Telemetry Service's own behaviour is covered by its suite. What matters
here is that the simulator produces events that service will accept, that a
seeded run is reproducible, and — most importantly — that it never invents a
verdict the model did not give.
"""

import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from simulator.events import (
    METRIC_NAMES,
    SCHEMA_VERSION,
    generate_event,
    summarize_response,
    vehicle_ids,
)
from simulator.run import parse_arguments

TAIPEI = ZoneInfo("Asia/Taipei")

ACCEPTED_UNITS = {
    "soc": {"percent", "fraction"},
    "battery_voltage": {"V", "mV"},
    "battery_current": {"A", "mA"},
    "battery_temperature": {"degC", "degF", "K"},
    "speed": {"km/h", "mph", "m/s"},
    "motor_rpm": {"rpm"},
}
"""The units the Telemetry Service accepts, per its committed contract."""


def make(rng: random.Random, *, injected: bool = False, vehicle: str = "EV-001", now=None):
    return generate_event(
        vehicle_id=vehicle,
        site_id="site-taipei-01",
        rng=rng,
        site_timezone=TAIPEI,
        inject_anomaly=injected,
        now=now,
    )


# --------------------------------------------------------------------------- #
# The fleet
# --------------------------------------------------------------------------- #


def test_the_default_fleet_is_named_as_documented() -> None:
    assert vehicle_ids(3) == ["EV-001", "EV-002", "EV-003"]


def test_a_larger_fleet_keeps_the_numbering() -> None:
    assert vehicle_ids(5)[-1] == "EV-005"


def test_an_empty_fleet_is_rejected() -> None:
    with pytest.raises(ValueError):
        vehicle_ids(0)


# --------------------------------------------------------------------------- #
# The payload matches the service's contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("injected", [False, True])
def test_the_payload_carries_exactly_the_contract_fields(injected: bool) -> None:
    payload = make(random.Random(1), injected=injected).payload

    assert set(payload) == {
        "schema_version",
        "event_id",
        "vehicle_id",
        "site_id",
        "event_time",
        "metrics",
    }
    assert payload["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("injected", [False, True])
def test_every_metric_is_present_with_a_value_and_an_accepted_unit(injected: bool) -> None:
    metrics = make(random.Random(2), injected=injected).payload["metrics"]

    assert set(metrics) == set(METRIC_NAMES)
    for name, measurement in metrics.items():
        assert set(measurement) == {"value", "unit"}
        assert isinstance(measurement["value"], float)
        assert measurement["unit"] in ACCEPTED_UNITS[name], name


def test_source_units_are_used_so_normalization_is_exercised() -> None:
    """mph and degF go out; the service is what converts them."""
    metrics = make(random.Random(3)).payload["metrics"]

    assert metrics["speed"]["unit"] == "mph"
    assert metrics["battery_temperature"]["unit"] == "degF"


def test_soc_and_speed_stay_inside_the_contract_bounds() -> None:
    """The service rejects SOC outside 0-100 and negative speed."""
    for seed in range(40):
        metrics = make(random.Random(seed)).payload["metrics"]
        assert 0.0 <= metrics["soc"]["value"] <= 100.0
        assert metrics["speed"]["value"] >= 0.0
        assert metrics["motor_rpm"]["value"] >= 0.0


def test_anomaly_candidates_also_respect_the_contract_bounds() -> None:
    """An anomaly must be *unusual*, not invalid — the service would reject invalid."""
    for seed in range(40):
        metrics = make(random.Random(seed), injected=True).payload["metrics"]
        assert 0.0 <= metrics["soc"]["value"] <= 100.0
        assert metrics["speed"]["value"] >= 0.0


# --------------------------------------------------------------------------- #
# Identity and time
# --------------------------------------------------------------------------- #


def test_every_event_gets_a_unique_event_id() -> None:
    rng = random.Random(4)
    ids = {make(rng).event_id for _ in range(200)}

    assert len(ids) == 200


def test_event_time_is_timezone_aware_with_the_sites_offset() -> None:
    """Naive timestamps are rejected by the service; the offset proves the point."""
    payload = make(random.Random(5)).payload

    parsed = datetime.fromisoformat(payload["event_time"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(hours=8)


def test_event_time_sits_just_behind_now_inside_the_clock_skew_window() -> None:
    """Comfortably inside the service's 300 s future bound and 30 day past bound."""
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)

    parsed = datetime.fromisoformat(make(random.Random(6), now=now).payload["event_time"])

    assert timedelta(seconds=0) < now - parsed < timedelta(seconds=5)


# --------------------------------------------------------------------------- #
# Determinism and injection
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_same_metrics() -> None:
    """Event ids are random by design, so the measurements are what must match."""
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)

    first = make(random.Random(11), now=now).payload
    second = make(random.Random(11), now=now).payload

    assert first["metrics"] == second["metrics"]
    assert first["event_time"] == second["event_time"]


def test_different_seeds_produce_different_metrics() -> None:
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)

    assert make(random.Random(11), now=now).payload["metrics"] != make(
        random.Random(12), now=now
    ).payload["metrics"]


def test_injection_is_forced_deterministically_not_left_to_chance() -> None:
    assert make(random.Random(7), injected=True).injected is True
    assert make(random.Random(7), injected=False).injected is False


def test_an_injected_event_is_far_from_normal_operation() -> None:
    """Not a validated fault signature — just clearly outside the training data."""
    rng = random.Random(8)
    anomalies = [make(rng, injected=True).payload["metrics"] for _ in range(20)]

    for metrics in anomalies:
        collapsed = metrics["battery_voltage"]["value"] < 300.0
        stationary_but_spinning = (
            metrics["speed"]["value"] < 1.0 and metrics["motor_rpm"]["value"] > 5000.0
        )
        assert collapsed or stationary_but_spinning


def test_the_intent_flag_never_reaches_the_service() -> None:
    """`injected` is the simulator's own bookkeeping, not part of the contract."""
    event = make(random.Random(9), injected=True)

    assert event.injected is True
    assert "injected" not in event.payload
    assert "anomaly" not in str(event.payload)


# --------------------------------------------------------------------------- #
# Reporting: the model stays authoritative
# --------------------------------------------------------------------------- #


def test_a_completed_verdict_is_reported_as_returned() -> None:
    line = summarize_response(
        201,
        {
            "duplicate": False,
            "inference": {
                "status": "COMPLETED",
                "is_anomaly": True,
                "anomaly_score": 0.1029,
                "error_code": None,
            },
        },
    )

    assert "COMPLETED" in line
    assert "anomaly=true" in line
    assert "+0.1029" in line


def test_a_failed_inference_reports_no_verdict_at_all() -> None:
    """The simulator must not fill in `false` where the model said nothing."""
    line = summarize_response(
        201,
        {
            "duplicate": False,
            "inference": {
                "status": "FAILED",
                "is_anomaly": None,
                "anomaly_score": None,
                "error_code": "INFERENCE_UNREACHABLE",
            },
        },
    )

    assert "FAILED" in line
    assert "INFERENCE_UNREACHABLE" in line
    assert "anomaly=—" in line
    assert "true" not in line and "false" not in line


def test_a_duplicate_is_reported_as_such() -> None:
    line = summarize_response(
        200,
        {"duplicate": True, "inference": {"status": "COMPLETED", "is_anomaly": False,
                                          "anomaly_score": -0.1, "error_code": None}},
    )

    assert "duplicate" in line


def test_an_error_envelope_is_reported_by_its_code() -> None:
    line = summarize_response(503, {"error": {"code": "PERSISTENCE_UNAVAILABLE"}})

    assert "503" in line
    assert "PERSISTENCE_UNAVAILABLE" in line


def test_a_missing_body_does_not_invent_a_result() -> None:
    assert "anomaly" not in summarize_response(500, None)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_the_defaults_target_the_local_compose_stack() -> None:
    arguments = parse_arguments([])

    assert arguments.url == "http://localhost:8000/api/v1/telemetry"
    assert arguments.vehicles == 3
    assert arguments.count == 0  # continuous


def test_options_override_the_defaults() -> None:
    arguments = parse_arguments(["--count", "5", "--seed", "42", "--anomaly-rate", "1.0"])

    assert (arguments.count, arguments.seed, arguments.anomaly_rate) == (5, 42, 1.0)
