"""The application boundary itself, exercised without any transport."""

from datetime import datetime, timedelta, timezone

import pytest

from app.application.errors import CapabilityNotImplementedError
from app.application.ingest_telemetry import IngestTelemetry
from app.domain.telemetry import Measurement, SourceTelemetryEvent
from app.domain.units import MetricName


def _event() -> SourceTelemetryEvent:
    return SourceTelemetryEvent(
        schema_version="1.0",
        event_id="3f0a9c2e-6f4b-4a6f-9d6e-2b1c8f4a77e1",
        vehicle_id="veh-tw-0142",
        site_id="site-taipei-01",
        event_time=datetime(2026, 8, 31, 9, 14, 22, tzinfo=timezone(timedelta(hours=8))),
        metrics={
            MetricName.SOC: Measurement(78.5, "percent"),
            MetricName.SPEED: Measurement(32.3, "mph"),
        },
    )


def test_use_case_is_callable_without_any_transport() -> None:
    """The use case takes a domain event, so MQTT could later call the same one."""
    with pytest.raises(CapabilityNotImplementedError) as raised:
        IngestTelemetry().execute(_event())

    assert raised.value.capability == "Telemetry ingestion"


def test_use_case_refuses_rather_than_returning_a_result() -> None:
    """Downstream does not exist, so accepting the event would be a lie."""
    with pytest.raises(CapabilityNotImplementedError):
        IngestTelemetry().execute(_event())


def test_use_case_needs_no_collaborators_in_this_phase() -> None:
    """Normalizer, repository and inference port are absent, not faked."""
    assert IngestTelemetry().__dict__ == {}
