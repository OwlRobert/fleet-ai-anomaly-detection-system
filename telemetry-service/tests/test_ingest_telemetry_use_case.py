"""The application boundary, exercised without any transport."""

from datetime import timedelta

import pytest

from app.application.errors import CapabilityNotImplementedError
from app.application.ingest_telemetry import IngestTelemetry
from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.errors import ClockSkewFutureError, EventTooOldError
from app.domain.normalizer import TelemetryNormalizer
from app.domain.telemetry import SourceTelemetryEvent
from app.domain.units import MetricName
from tests.factories import FIXED_NOW, one_metric, source_event


def use_case() -> IngestTelemetry:
    return IngestTelemetry(normalizer=TelemetryNormalizer(clock=lambda: FIXED_NOW))


class RecordingNormalizer(TelemetryNormalizer):
    """Records what it was handed, then normalizes for real."""

    def __init__(self) -> None:
        super().__init__(clock=lambda: FIXED_NOW)
        self.seen: list[SourceTelemetryEvent] = []
        self.produced: list[CanonicalTelemetryEvent] = []

    def normalize(self, source_event: SourceTelemetryEvent) -> CanonicalTelemetryEvent:
        self.seen.append(source_event)
        canonical = super().normalize(source_event)
        self.produced.append(canonical)
        return canonical


def test_use_case_is_callable_without_any_transport() -> None:
    """It takes a domain event, so MQTT could later call the same one."""
    with pytest.raises(CapabilityNotImplementedError) as raised:
        use_case().execute(source_event())

    assert raised.value.capability == "Telemetry ingestion"


def test_the_source_event_reaches_the_normalizer_unchanged() -> None:
    normalizer = RecordingNormalizer()
    event = source_event(metrics=one_metric(MetricName.SPEED, 32.3, "mph"))

    with pytest.raises(CapabilityNotImplementedError):
        IngestTelemetry(normalizer=normalizer).execute(event)

    assert normalizer.seen == [event]


def test_normalization_actually_runs_before_the_refusal() -> None:
    """The refusal must come *after* real work, not instead of it."""
    normalizer = RecordingNormalizer()
    event = source_event(metrics=one_metric(MetricName.SPEED, 32.3, "mph"))

    with pytest.raises(CapabilityNotImplementedError):
        IngestTelemetry(normalizer=normalizer).execute(event)

    canonical = normalizer.produced[0]
    assert canonical.metrics[MetricName.SPEED] == pytest.approx(51.9818112)
    assert canonical.source_units[MetricName.SPEED] == "mph"
    assert canonical.received_at == FIXED_NOW


def test_refusal_names_what_is_still_missing() -> None:
    """Normalization is done; inference and persistence are not."""
    with pytest.raises(CapabilityNotImplementedError) as raised:
        use_case().execute(source_event())

    assert "normalized" in raised.value.arrives_in
    assert "inference and persistence" in raised.value.arrives_in


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        pytest.param(timedelta(days=1), ClockSkewFutureError, id="too-far-ahead"),
        pytest.param(timedelta(days=-90), EventTooOldError, id="too-old"),
    ],
)
def test_normalization_errors_propagate_instead_of_the_not_implemented_refusal(
    offset: timedelta, expected: type[Exception]
) -> None:
    """A rejected event never reaches the point where the pipeline stops."""
    with pytest.raises(expected):
        use_case().execute(source_event(event_time=FIXED_NOW + offset))


def test_use_case_holds_only_the_collaborators_that_exist() -> None:
    """No repository and no inference port are faked to fill the pipeline."""
    instance = use_case()

    assert list(vars(instance)) == ["_normalizer"]
