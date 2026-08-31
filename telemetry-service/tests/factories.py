"""Builders for domain events used across the Phase 2 tests."""

from datetime import datetime, timezone
from typing import Mapping

from app.domain.telemetry import Measurement, SourceTelemetryEvent
from app.domain.units import MetricName

FIXED_NOW = datetime(2026, 9, 1, 16, 0, 0, tzinfo=timezone.utc)
"""The pinned server clock the normalizer tests run against."""

CANONICAL_SOURCE_METRICS: Mapping[MetricName, Measurement] = {
    MetricName.SOC: Measurement(78.5, "percent"),
    MetricName.BATTERY_VOLTAGE: Measurement(396.2, "V"),
    MetricName.BATTERY_CURRENT: Measurement(-14.7, "A"),
    MetricName.BATTERY_TEMPERATURE: Measurement(35.0, "degC"),
    MetricName.SPEED: Measurement(51.9, "km/h"),
    MetricName.MOTOR_RPM: Measurement(4120.0, "rpm"),
}
"""Every metric already in its canonical unit, so conversions are pass-through."""


def source_event(
    *,
    event_time: datetime | None = None,
    metrics: Mapping[MetricName, Measurement] | None = None,
    event_id: str = "3f0a9c2e-6f4b-4a6f-9d6e-2b1c8f4a77e1",
    vehicle_id: str = "veh-tw-0142",
    site_id: str = "site-taipei-01",
    schema_version: str = "1.0",
) -> SourceTelemetryEvent:
    """A valid source event, with each field overridable."""
    return SourceTelemetryEvent(
        schema_version=schema_version,
        event_id=event_id,
        vehicle_id=vehicle_id,
        site_id=site_id,
        event_time=event_time if event_time is not None else FIXED_NOW,
        metrics=dict(metrics if metrics is not None else CANONICAL_SOURCE_METRICS),
    )


def one_metric(metric: MetricName, value: float, unit: str) -> Mapping[MetricName, Measurement]:
    """Canonical metrics with a single metric replaced."""
    return {**CANONICAL_SOURCE_METRICS, metric: Measurement(value, unit)}
