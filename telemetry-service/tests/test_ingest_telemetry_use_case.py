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
from tests.fakes import InMemoryTelemetryRepository, UnavailableTelemetryRepository

pytestmark = pytest.mark.anyio


def use_case(repository=None) -> IngestTelemetry:
    return IngestTelemetry(
        normalizer=TelemetryNormalizer(clock=lambda: FIXED_NOW),
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


async def test_a_stored_event_is_unscored() -> None:
    """No model ran, so no verdict is invented."""
    repository = InMemoryTelemetryRepository()

    await use_case(repository).execute(source_event())

    inference = repository.events[0].inference
    assert inference.status is InferenceStatus.PENDING
    assert inference.is_anomaly is None
    assert inference.anomaly_score is None
    assert inference.model_name is None
    assert inference.model_version is None
    assert inference.error_code is None
    assert inference.is_confirmed_anomaly is False


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


async def test_use_case_holds_only_the_collaborators_that_exist() -> None:
    """No inference port is faked to fill the pipeline."""
    assert sorted(vars(use_case())) == ["_normalizer", "_repository"]
