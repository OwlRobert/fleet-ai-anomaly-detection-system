"""The application boundary, exercised without any transport."""

from datetime import timedelta

import pytest

from app.application.errors import PersistenceUnavailableError
from app.application.ingest_telemetry import IngestTelemetry
from app.domain.errors import ClockSkewFutureError, EventTooOldError
from app.domain.inference import InferenceStatus
from app.domain.normalizer import TelemetryNormalizer
from app.domain.units import MetricName
from tests.factories import FIXED_NOW, one_metric, source_event
from tests.fakes import (
    InMemoryTelemetryRepository,
    StubInferencePort,
    UnavailableTelemetryRepository,
)

pytestmark = pytest.mark.anyio


def use_case(repository=None, inference=None) -> IngestTelemetry:
    return IngestTelemetry(
        normalizer=TelemetryNormalizer(clock=lambda: FIXED_NOW),
        inference=inference if inference is not None else StubInferencePort(),
        repository=repository if repository is not None else InMemoryTelemetryRepository(),
    )


async def test_use_case_is_callable_without_any_transport() -> None:
    """It takes a domain event, so MQTT could later call the same one."""
    outcome = await use_case().execute(source_event())

    assert outcome.duplicate is False


async def test_normalization_runs_before_persistence() -> None:
    repository = InMemoryTelemetryRepository()

    await use_case(repository).execute(source_event(metrics=one_metric(MetricName.SPEED, 32.3, "mph")))

    stored = repository.events[0]
    assert stored.event.metrics[MetricName.SPEED] == pytest.approx(51.9818112)
    assert stored.event.source_units[MetricName.SPEED] == "mph"
    assert stored.event.received_at == FIXED_NOW


async def test_a_stored_event_carries_the_models_verdict() -> None:
    """The model ran, so its answer is what gets stored."""
    repository = InMemoryTelemetryRepository()
    model = StubInferencePort(is_anomaly=True, anomaly_score=0.1029)

    await use_case(repository, model).execute(source_event())

    inference = repository.events[0].inference
    assert inference.status is InferenceStatus.COMPLETED
    assert inference.is_anomaly is True
    assert inference.anomaly_score == 0.1029
    assert inference.model_name == "isolation-forest-telemetry"
    assert inference.model_version == "0.1.0"
    assert inference.error_code is None
    assert inference.is_confirmed_anomaly is True


async def test_a_synchronous_ingest_never_finishes_pending() -> None:
    """PENDING was the honest state before inference existed. It is not terminal now."""
    repository = InMemoryTelemetryRepository()

    await use_case(repository).execute(source_event())

    assert repository.events[0].inference.status is not InferenceStatus.PENDING


async def test_transport_provenance_is_recorded() -> None:
    repository = InMemoryTelemetryRepository()

    await use_case(repository).execute(source_event(), transport="rest", api_version="v1")

    assert repository.events[0].transport == "rest"
    assert repository.events[0].api_version == "v1"


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        pytest.param(timedelta(days=1), ClockSkewFutureError, id="too-far-ahead"),
        pytest.param(timedelta(days=-90), EventTooOldError, id="too-old"),
    ],
)
async def test_a_rejected_event_is_never_stored(offset: timedelta, expected: type[Exception]) -> None:
    repository = InMemoryTelemetryRepository()

    with pytest.raises(expected):
        await use_case(repository).execute(source_event(event_time=FIXED_NOW + offset))

    assert repository.events == []


async def test_persistence_failure_propagates_so_ingestion_fails_closed() -> None:
    """The caller must never be told an unstored event was accepted."""
    with pytest.raises(PersistenceUnavailableError):
        await use_case(UnavailableTelemetryRepository()).execute(source_event())


async def test_use_case_holds_exactly_its_collaborators() -> None:
    assert sorted(vars(use_case())) == ["_inference", "_normalizer", "_repository"]
