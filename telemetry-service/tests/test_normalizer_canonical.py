"""The canonical event the normalizer produces."""

import ast
import pathlib
import sys
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import pytest

import app.domain
from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.normalizer import TelemetryNormalizer
from app.domain.telemetry import Measurement
from app.domain.units import CANONICAL_UNITS, MetricName
from tests.factories import CANONICAL_SOURCE_METRICS, FIXED_NOW, one_metric, source_event

MIXED_UNIT_METRICS = {
    MetricName.SOC: Measurement(0.725, "fraction"),
    MetricName.BATTERY_VOLTAGE: Measurement(396200.0, "mV"),
    MetricName.BATTERY_CURRENT: Measurement(-14700.0, "mA"),
    MetricName.BATTERY_TEMPERATURE: Measurement(96.4, "degF"),
    MetricName.SPEED: Measurement(32.3, "mph"),
    MetricName.MOTOR_RPM: Measurement(4120.0, "rpm"),
}
"""A device mixing unit systems within one payload, which is the normal case."""


def normalizer() -> TelemetryNormalizer:
    return TelemetryNormalizer(clock=lambda: FIXED_NOW)


# --------------------------------------------------------------------------- #
# Identity and source metadata carried through unchanged
# --------------------------------------------------------------------------- #


def test_identity_and_metadata_are_preserved_exactly() -> None:
    event = source_event(
        event_id="site-taipei-01:seq-000917",
        vehicle_id="veh-tw-0142",
        site_id="site-taipei-01",
        schema_version="1.0",
        event_time=FIXED_NOW - timedelta(minutes=5),
    )

    canonical = normalizer().normalize(event)

    assert canonical.event_id == "site-taipei-01:seq-000917"
    assert canonical.vehicle_id == "veh-tw-0142"
    assert canonical.site_id == "site-taipei-01"
    assert canonical.schema_version == "1.0"


def test_event_id_is_carried_through_opaquely() -> None:
    """The normalizer neither parses nor rewrites the domain identifier."""
    canonical = normalizer().normalize(source_event(event_id="  odd/but/valid  "))

    assert canonical.event_id == "  odd/but/valid  "


# --------------------------------------------------------------------------- #
# Canonical metrics
# --------------------------------------------------------------------------- #


def test_all_metrics_are_converted_to_canonical_values() -> None:
    canonical = normalizer().normalize(source_event(metrics=MIXED_UNIT_METRICS))

    assert canonical.metrics[MetricName.SOC] == pytest.approx(72.5)
    assert canonical.metrics[MetricName.BATTERY_VOLTAGE] == pytest.approx(396.2)
    assert canonical.metrics[MetricName.BATTERY_CURRENT] == pytest.approx(-14.7)
    assert canonical.metrics[MetricName.BATTERY_TEMPERATURE] == pytest.approx(35.7777777777)
    assert canonical.metrics[MetricName.SPEED] == pytest.approx(51.9818112)
    assert canonical.metrics[MetricName.MOTOR_RPM] == pytest.approx(4120.0)


def test_canonical_metrics_are_bare_numbers_without_units() -> None:
    """The unit is a property of the schema, never of each value."""
    canonical = normalizer().normalize(source_event(metrics=MIXED_UNIT_METRICS))

    assert set(canonical.metrics) == set(MetricName)
    for value in canonical.metrics.values():
        assert isinstance(value, float)


def test_no_source_unit_value_leaks_into_the_canonical_metrics() -> None:
    """32.3 mph must not survive as 32.3; it becomes 51.98 km/h."""
    canonical = normalizer().normalize(source_event(metrics=MIXED_UNIT_METRICS))

    assert canonical.metrics[MetricName.SPEED] != pytest.approx(32.3)
    assert canonical.metrics[MetricName.BATTERY_TEMPERATURE] != pytest.approx(96.4)
    assert canonical.metrics[MetricName.SOC] != pytest.approx(0.725)


def test_already_canonical_metrics_pass_through_unchanged() -> None:
    canonical = normalizer().normalize(source_event(metrics=CANONICAL_SOURCE_METRICS))

    for metric, measurement in CANONICAL_SOURCE_METRICS.items():
        assert canonical.metrics[metric] == pytest.approx(measurement.value)


# --------------------------------------------------------------------------- #
# Source-unit provenance
# --------------------------------------------------------------------------- #


def test_source_units_are_kept_as_provenance() -> None:
    canonical = normalizer().normalize(source_event(metrics=MIXED_UNIT_METRICS))

    assert canonical.source_units == {
        MetricName.SOC: "fraction",
        MetricName.BATTERY_VOLTAGE: "mV",
        MetricName.BATTERY_CURRENT: "mA",
        MetricName.BATTERY_TEMPERATURE: "degF",
        MetricName.SPEED: "mph",
        MetricName.MOTOR_RPM: "rpm",
    }


def test_provenance_records_the_unit_not_a_duplicate_value() -> None:
    """Provenance is compact metadata, not a second copy of the measurement."""
    canonical = normalizer().normalize(source_event(metrics=MIXED_UNIT_METRICS))

    for unit in canonical.source_units.values():
        assert isinstance(unit, str)


def test_provenance_is_recorded_even_when_the_source_unit_is_canonical() -> None:
    canonical = normalizer().normalize(source_event(metrics=CANONICAL_SOURCE_METRICS))

    for metric, unit in canonical.source_units.items():
        assert unit == CANONICAL_UNITS[metric]


# --------------------------------------------------------------------------- #
# The source event is never mutated
# --------------------------------------------------------------------------- #


def test_the_source_event_is_left_untouched() -> None:
    event = source_event(
        metrics=one_metric(MetricName.SPEED, 32.3, "mph"),
        event_time=datetime(2026, 9, 1, 12, 30, tzinfo=timezone(timedelta(hours=8))),
    )

    normalizer().normalize(event)

    assert event.metrics[MetricName.SPEED] == Measurement(32.3, "mph")
    assert event.event_time.utcoffset() == timedelta(hours=8)


def test_normalization_returns_a_new_object() -> None:
    event = source_event()

    canonical = normalizer().normalize(event)

    assert isinstance(canonical, CanonicalTelemetryEvent)
    assert canonical is not event


def test_canonical_metrics_cannot_be_mutated_after_construction() -> None:
    canonical = normalizer().normalize(source_event())

    with pytest.raises(TypeError):
        canonical.metrics[MetricName.SPEED] = 0.0  # type: ignore[index]


# --------------------------------------------------------------------------- #
# What the canonical model must NOT contain
# --------------------------------------------------------------------------- #


def test_canonical_model_holds_no_storage_or_inference_concepts() -> None:
    canonical = normalizer().normalize(source_event())

    assert {field.name for field in fields(CanonicalTelemetryEvent)} == {
        "schema_version",
        "event_id",
        "vehicle_id",
        "site_id",
        "event_time",
        "received_at",
        "metrics",
        "source_units",
    }
    for absent in ("_id", "id", "created_at", "inference", "is_anomaly", "anomaly_score"):
        assert not hasattr(canonical, absent)


def test_canonical_model_rejects_a_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        CanonicalTelemetryEvent(
            schema_version="1.0",
            event_id="e1",
            vehicle_id="v1",
            site_id="s1",
            event_time=datetime(2026, 9, 1, 4, 30),
            received_at=FIXED_NOW,
            metrics={},
            source_units={},
        )


def test_canonical_model_rejects_a_non_utc_timestamp() -> None:
    with pytest.raises(ValueError, match="normalized to UTC"):
        CanonicalTelemetryEvent(
            schema_version="1.0",
            event_id="e1",
            vehicle_id="v1",
            site_id="s1",
            event_time=datetime(2026, 9, 1, 12, 30, tzinfo=timezone(timedelta(hours=8))),
            received_at=FIXED_NOW,
            metrics={},
            source_units={},
        )


FORBIDDEN_IN_DOMAIN = {
    "fastapi", "starlette", "pydantic", "pydantic_settings",
    "pymongo", "motor", "bson", "sqlalchemy", "psycopg",
    "sklearn", "joblib", "numpy", "httpx", "requests",
}


def test_domain_imports_no_framework_database_or_ml_package() -> None:
    """The domain stays pure: only the standard library and `app.domain`."""
    domain_dir = pathlib.Path(app.domain.__file__).parent

    for module_path in sorted(domain_dir.glob("*.py")):
        tree = ast.parse(module_path.read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & FORBIDDEN_IN_DOMAIN), (module_path.name, imported)
        assert imported <= {"app"} | set(sys.stdlib_module_names), (module_path.name, imported)
