"""Builders for domain events used across the Phase 2 tests."""

from datetime import datetime, timedelta, timezone
from typing import Mapping

from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.inference import InferenceOutcome, InferenceStatus
from app.domain.stored import StoredTelemetryEvent
from app.domain.telemetry import Measurement, SourceTelemetryEvent
from app.domain.units import CANONICAL_UNITS, MetricName

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


CANONICAL_METRICS = {
    MetricName.SOC: 78.5,
    MetricName.BATTERY_VOLTAGE: 396.2,
    MetricName.BATTERY_CURRENT: -14.7,
    MetricName.BATTERY_TEMPERATURE: 35.0,
    MetricName.SPEED: 51.9,
    MetricName.MOTOR_RPM: 4120.0,
}


def canonical_event(
    *,
    event_id: str = "3f0a9c2e-6f4b-4a6f-9d6e-2b1c8f4a77e1",
    vehicle_id: str = "veh-tw-0142",
    site_id: str = "site-taipei-01",
    schema_version: str = "1.0",
    event_time: datetime | None = None,
    received_at: datetime | None = None,
    metrics: dict[MetricName, float] | None = None,
    source_units: dict[MetricName, str] | None = None,
) -> CanonicalTelemetryEvent:
    """A canonical event, with each field overridable."""
    return CanonicalTelemetryEvent(
        schema_version=schema_version,
        event_id=event_id,
        vehicle_id=vehicle_id,
        site_id=site_id,
        event_time=event_time if event_time is not None else FIXED_NOW - timedelta(minutes=5),
        received_at=received_at if received_at is not None else FIXED_NOW,
        metrics=dict(metrics if metrics is not None else CANONICAL_METRICS),
        source_units=dict(source_units if source_units is not None else CANONICAL_UNITS),
    )


def stored_event(
    *,
    inference: InferenceOutcome | None = None,
    transport: str = "rest",
    api_version: str = "v1",
    **event_kwargs,
) -> StoredTelemetryEvent:
    """A stored event. Unscored by default, which is the Phase 3 reality."""
    return StoredTelemetryEvent(
        event=canonical_event(**event_kwargs),
        inference=inference if inference is not None else InferenceOutcome.pending(),
        transport=transport,
        api_version=api_version,
    )


def scored(is_anomaly: bool) -> InferenceOutcome:
    """A completed inference verdict.

    This is a **storage fixture** describing the document shape a future scored
    event will have. No model ran; nothing here is an ML result.
    """
    return InferenceOutcome(
        status=InferenceStatus.COMPLETED,
        is_anomaly=is_anomaly,
        anomaly_score=-0.31 if is_anomaly else 0.12,
        model_name="isolation-forest-telemetry",
        model_version="0.1.0",
    )
