"""Idempotent ingestion, and what happens when an event_id is reused.

The unique constraint in the store is the correctness boundary; the lookup
before the write is only an optimization. Both paths are exercised.
"""

from datetime import timedelta

import pytest

from app.application.errors import DuplicateEventIdError, PersistenceUnavailableError
from app.application.ingest_telemetry import IngestTelemetry
from app.domain.normalizer import TelemetryNormalizer
from app.domain.units import MetricName
from tests.factories import FIXED_NOW, one_metric, source_event
from tests.fakes import InMemoryTelemetryRepository

pytestmark = pytest.mark.anyio

EVENT_ID = "3f0a9c2e-6f4b-4a6f-9d6e-2b1c8f4a77e1"


def use_case(repository: InMemoryTelemetryRepository, now=FIXED_NOW) -> IngestTelemetry:
    return IngestTelemetry(
        normalizer=TelemetryNormalizer(clock=lambda: now), repository=repository
    )


# --------------------------------------------------------------------------- #
# First write, and the exact retry
# --------------------------------------------------------------------------- #


async def test_a_first_event_creates_exactly_one_record() -> None:
    repository = InMemoryTelemetryRepository()

    outcome = await use_case(repository).execute(source_event())

    assert outcome.duplicate is False
    assert outcome.conflict is False
    assert len(repository.events) == 1


async def test_an_exact_retry_reports_a_duplicate() -> None:
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event())

    outcome = await use_case(repository).execute(source_event())

    assert outcome.duplicate is True
    assert outcome.conflict is False


async def test_an_exact_retry_creates_no_second_record() -> None:
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event())
    await use_case(repository).execute(source_event())
    await use_case(repository).execute(source_event())

    assert len(repository.events) == 1


async def test_an_exact_retry_does_not_overwrite_the_original() -> None:
    """The retry arrives later, so its received_at differs. The first one wins."""
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event())
    later = FIXED_NOW + timedelta(minutes=9)

    outcome = await use_case(repository, now=later).execute(source_event())

    assert outcome.stored.event.received_at == FIXED_NOW
    assert repository.events[0].event.received_at == FIXED_NOW


async def test_a_retry_returns_the_event_that_was_stored_first() -> None:
    repository = InMemoryTelemetryRepository()
    first = await use_case(repository).execute(source_event())

    retry = await use_case(repository, now=FIXED_NOW + timedelta(hours=1)).execute(source_event())

    assert retry.stored == first.stored


# --------------------------------------------------------------------------- #
# Conflicting reuse of an event_id
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"vehicle_id": "veh-cz-0007"}, id="different-vehicle"),
        pytest.param({"site_id": "site-prague-02"}, id="different-site"),
        pytest.param({"event_time": FIXED_NOW - timedelta(hours=3)}, id="different-event-time"),
        pytest.param({"schema_version": "1.0 "}, id="different-schema-version"),
    ],
)
async def test_reusing_an_event_id_with_different_content_conflicts(mutation: dict) -> None:
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event(event_id=EVENT_ID))

    outcome = await use_case(repository).execute(source_event(event_id=EVENT_ID, **mutation))

    assert outcome.conflict is True
    assert outcome.duplicate is True


async def test_reusing_an_event_id_with_different_metrics_conflicts() -> None:
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event(event_id=EVENT_ID))

    outcome = await use_case(repository).execute(
        source_event(event_id=EVENT_ID, metrics=one_metric(MetricName.SPEED, 99.9, "km/h"))
    )

    assert outcome.conflict is True


async def test_a_conflicting_retry_never_overwrites_the_original() -> None:
    """First write wins. The stored event is returned untouched."""
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event(event_id=EVENT_ID, vehicle_id="veh-tw-0142"))

    outcome = await use_case(repository).execute(
        source_event(event_id=EVENT_ID, vehicle_id="veh-cz-0007")
    )

    assert len(repository.events) == 1
    assert repository.events[0].event.vehicle_id == "veh-tw-0142"
    assert outcome.stored.event.vehicle_id == "veh-tw-0142"


async def test_a_conflict_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    repository = InMemoryTelemetryRepository()
    await use_case(repository).execute(source_event(event_id=EVENT_ID))

    with caplog.at_level("WARNING"):
        await use_case(repository).execute(source_event(event_id=EVENT_ID, vehicle_id="veh-cz-0007"))

    assert "event_id reused with different content" in caplog.text


# --------------------------------------------------------------------------- #
# The race the pre-write lookup cannot close
# --------------------------------------------------------------------------- #


async def test_a_lost_race_with_an_equivalent_event_resolves_as_a_duplicate() -> None:
    """Both requests saw 'absent'; the unique constraint decided the winner."""
    repository = InMemoryTelemetryRepository()
    competitor = source_event(event_id=EVENT_ID)

    def concurrent_writer() -> None:
        repository.before_save = None
        repository.events.append(_stored(repository, competitor))

    repository.before_save = concurrent_writer

    outcome = await use_case(repository).execute(source_event(event_id=EVENT_ID))

    assert outcome.duplicate is True
    assert outcome.conflict is False
    assert len(repository.events) == 1


async def test_a_lost_race_with_a_conflicting_event_resolves_as_a_conflict() -> None:
    repository = InMemoryTelemetryRepository()
    competitor = source_event(event_id=EVENT_ID, vehicle_id="veh-cz-0007")

    def concurrent_writer() -> None:
        repository.before_save = None
        repository.events.append(_stored(repository, competitor))

    repository.before_save = concurrent_writer

    outcome = await use_case(repository).execute(source_event(event_id=EVENT_ID))

    assert outcome.conflict is True
    assert len(repository.events) == 1
    assert repository.events[0].event.vehicle_id == "veh-cz-0007"


async def test_correctness_does_not_rely_on_the_pre_write_lookup() -> None:
    """With the lookup blind, the unique constraint still prevents a second record."""
    repository = InMemoryTelemetryRepository()

    async def blind_lookup(event_id: str):
        return None

    await use_case(repository).execute(source_event(event_id=EVENT_ID))
    repository.find_by_event_id = blind_lookup  # type: ignore[method-assign]

    with pytest.raises(DuplicateEventIdError):
        await repository.save(_stored(repository, source_event(event_id=EVENT_ID)))

    assert len(repository.events) == 1


async def test_a_duplicate_whose_event_vanishes_fails_closed() -> None:
    """Reported duplicate but nothing readable back: never claim success."""
    repository = InMemoryTelemetryRepository()

    async def absent(event_id: str):
        return None

    async def always_duplicate(event) -> None:
        raise DuplicateEventIdError(event.event_id)

    repository.find_by_event_id = absent  # type: ignore[method-assign]
    repository.save = always_duplicate  # type: ignore[method-assign]

    with pytest.raises(PersistenceUnavailableError):
        await use_case(repository).execute(source_event())


def _stored(repository: InMemoryTelemetryRepository, event):
    """Normalize an event into the stored shape a competitor would have written."""
    from app.domain.inference import InferenceOutcome
    from app.domain.stored import StoredTelemetryEvent

    canonical = TelemetryNormalizer(clock=lambda: FIXED_NOW).normalize(event)
    return StoredTelemetryEvent(
        event=canonical,
        inference=InferenceOutcome.pending(),
        transport="rest",
        api_version="v1",
    )
