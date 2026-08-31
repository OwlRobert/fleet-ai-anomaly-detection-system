"""Integration tests against a real MongoDB.

These are the only tests that prove MongoDB actually behaves as the design
assumes — that the unique index rejects a second insert, that BSON round-trips
the UTC instant, that the sort and range operators mean what the repository
thinks they mean. The in-memory double cannot prove any of that.

They run only when ``TEST_MONGODB_URI`` is set, and they use their own database,
which is dropped afterwards:

    TEST_MONGODB_URI=mongodb://localhost:27017 python -m pytest tests/test_mongodb_integration.py

Without it the whole module is skipped, and the skip reason says so rather than
quietly reporting success.
"""

import os
from datetime import timedelta, timezone

import pytest

from app.application.errors import DuplicateEventIdError
from app.domain.inference import InferenceStatus
from app.domain.units import MetricName
from app.infrastructure.mongo import (
    ANOMALY_INDEX,
    EVENT_ID_UNIQUE_INDEX,
    VEHICLE_TIME_INDEX,
    create_client,
    ensure_indexes,
)
from app.infrastructure.telemetry_repository import MongoTelemetryRepository
from tests.factories import FIXED_NOW, scored, stored_event

TEST_URI = os.environ.get("TEST_MONGODB_URI")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not TEST_URI,
        reason="TEST_MONGODB_URI is not set; no MongoDB available for integration tests",
    ),
]

TEST_DATABASE = "fleet_telemetry_integration_test"


@pytest.fixture
async def collection():
    """A dedicated, indexed collection, dropped after each test."""
    client = create_client(TEST_URI, timeout_ms=5000)
    database = client[TEST_DATABASE]
    telemetry = database["telemetry_events"]
    await telemetry.drop()
    await ensure_indexes(telemetry)
    try:
        yield telemetry
    finally:
        await client.drop_database(TEST_DATABASE)
        await client.close()


@pytest.fixture
def repository(collection) -> MongoTelemetryRepository:
    return MongoTelemetryRepository(collection)


# --------------------------------------------------------------------------- #
# Indexes
# --------------------------------------------------------------------------- #


async def test_every_declared_index_exists(collection) -> None:
    information = await collection.index_information()

    assert {EVENT_ID_UNIQUE_INDEX, VEHICLE_TIME_INDEX, ANOMALY_INDEX} <= set(information)


async def test_no_index_is_created_beyond_those_declared(collection) -> None:
    """Only MongoDB's own _id index accompanies the three declared ones."""
    information = await collection.index_information()

    assert set(information) == {
        "_id_",
        EVENT_ID_UNIQUE_INDEX,
        VEHICLE_TIME_INDEX,
        ANOMALY_INDEX,
    }


async def test_the_event_id_index_is_unique(collection) -> None:
    information = await collection.index_information()

    assert information[EVENT_ID_UNIQUE_INDEX].get("unique") is True
    assert information[EVENT_ID_UNIQUE_INDEX]["key"] == [("event_id", 1)]


async def test_the_anomaly_index_is_partial(collection) -> None:
    information = await collection.index_information()

    assert information[ANOMALY_INDEX]["partialFilterExpression"] == {
        "inference.is_anomaly": True,
        "inference.status": "COMPLETED",
    }


async def test_ensure_indexes_can_run_twice(collection) -> None:
    before = set(await collection.index_information())

    await ensure_indexes(collection)

    assert set(await collection.index_information()) == before


# --------------------------------------------------------------------------- #
# Insert and uniqueness
# --------------------------------------------------------------------------- #


async def test_an_event_can_be_stored_and_read_back(repository) -> None:
    original = stored_event(event_id="evt-1")

    await repository.save(original)

    assert await repository.find_by_event_id("evt-1") == original


async def test_the_unique_index_rejects_a_second_insert(repository) -> None:
    """This is the correctness boundary for idempotency, proven against MongoDB."""
    await repository.save(stored_event(event_id="evt-1"))

    with pytest.raises(DuplicateEventIdError):
        await repository.save(stored_event(event_id="evt-1", vehicle_id="veh-cz-0007"))


async def test_a_rejected_duplicate_leaves_the_original_untouched(repository, collection) -> None:
    await repository.save(stored_event(event_id="evt-1", vehicle_id="veh-tw-0142"))

    with pytest.raises(DuplicateEventIdError):
        await repository.save(stored_event(event_id="evt-1", vehicle_id="veh-cz-0007"))

    assert await collection.count_documents({}) == 1
    stored = await repository.find_by_event_id("evt-1")
    assert stored.event.vehicle_id == "veh-tw-0142"


async def test_mongodb_assigns_its_own_storage_identity(repository, collection) -> None:
    """_id exists in the document but never reaches the domain object."""
    await repository.save(stored_event(event_id="evt-1"))

    document = await collection.find_one({"event_id": "evt-1"})
    assert "_id" in document

    stored = await repository.find_by_event_id("evt-1")
    assert not hasattr(stored, "_id")
    assert stored.event_id == "evt-1"


async def test_missing_events_read_back_as_none(repository) -> None:
    assert await repository.find_by_event_id("never-stored") is None


# --------------------------------------------------------------------------- #
# Datetime fidelity
# --------------------------------------------------------------------------- #


async def test_timestamps_round_trip_as_aware_utc(repository) -> None:
    event_time = FIXED_NOW - timedelta(hours=3, minutes=17)
    await repository.save(stored_event(event_id="evt-1", event_time=event_time))

    stored = await repository.find_by_event_id("evt-1")

    assert stored.event.event_time == event_time
    assert stored.event.event_time.tzinfo is not None
    assert stored.event.event_time.utcoffset() == timedelta(0)
    assert stored.event.received_at.utcoffset() == timedelta(0)


async def test_a_source_offset_is_stored_as_the_same_instant(repository) -> None:
    """MongoDB keeps the instant; the offset representation is not preserved."""
    taipei = FIXED_NOW.astimezone(timezone(timedelta(hours=8)))
    await repository.save(stored_event(event_id="evt-1", event_time=taipei.astimezone(timezone.utc)))

    stored = await repository.find_by_event_id("evt-1")

    assert stored.event.event_time == taipei


async def test_canonical_metrics_and_provenance_survive_storage(repository) -> None:
    await repository.save(
        stored_event(event_id="evt-1", source_units={metric: "mph" for metric in MetricName})
    )

    stored = await repository.find_by_event_id("evt-1")

    assert stored.event.metrics[MetricName.SPEED] == pytest.approx(51.9)
    assert stored.event.source_units[MetricName.SPEED] == "mph"


async def test_an_unscored_event_reads_back_as_pending(repository) -> None:
    await repository.save(stored_event(event_id="evt-1"))

    stored = await repository.find_by_event_id("evt-1")

    assert stored.inference.status is InferenceStatus.PENDING
    assert stored.inference.is_anomaly is None


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


async def _seed(repository, events) -> None:
    for event in events:
        await repository.save(event)


async def test_history_is_ordered_newest_event_time_first(repository) -> None:
    await _seed(
        repository,
        [
            stored_event(event_id="oldest", event_time=FIXED_NOW - timedelta(hours=9)),
            stored_event(event_id="newest", event_time=FIXED_NOW - timedelta(hours=1)),
            stored_event(event_id="middle", event_time=FIXED_NOW - timedelta(hours=5)),
        ],
    )

    found = await repository.find_by_vehicle_and_time_range(
        "veh-tw-0142", FIXED_NOW - timedelta(days=1), FIXED_NOW, 100
    )

    assert [stored.event_id for stored in found] == ["newest", "middle", "oldest"]


async def test_the_range_is_half_open(repository) -> None:
    start, end = FIXED_NOW - timedelta(hours=4), FIXED_NOW - timedelta(hours=1)
    await _seed(
        repository,
        [
            stored_event(event_id="before", event_time=start - timedelta(seconds=1)),
            stored_event(event_id="on-start", event_time=start),
            stored_event(event_id="on-end", event_time=end),
        ],
    )

    found = await repository.find_by_vehicle_and_time_range("veh-tw-0142", start, end, 100)

    assert {stored.event_id for stored in found} == {"on-start"}


async def test_queries_filter_by_vehicle(repository) -> None:
    await _seed(
        repository,
        [
            stored_event(event_id="ours", vehicle_id="veh-tw-0142"),
            stored_event(event_id="theirs", vehicle_id="veh-cz-0007"),
        ],
    )

    found = await repository.find_by_vehicle_and_time_range(
        "veh-tw-0142", FIXED_NOW - timedelta(days=1), FIXED_NOW + timedelta(days=1), 100
    )

    assert [stored.event_id for stored in found] == ["ours"]


async def test_the_limit_is_applied(repository) -> None:
    await _seed(
        repository,
        [
            stored_event(event_id=f"evt-{index}", event_time=FIXED_NOW - timedelta(minutes=index))
            for index in range(1, 6)
        ],
    )

    found = await repository.find_by_vehicle_and_time_range(
        "veh-tw-0142", FIXED_NOW - timedelta(days=1), FIXED_NOW + timedelta(days=1), 2
    )

    assert [stored.event_id for stored in found] == ["evt-1", "evt-2"]


async def test_anomalies_exclude_unscored_and_normal_events(repository) -> None:
    """Storage fixtures for future scored documents. No model ran."""
    await _seed(
        repository,
        [
            stored_event(event_id="anomalous", inference=scored(is_anomaly=True)),
            stored_event(event_id="scored-normal", inference=scored(is_anomaly=False)),
            stored_event(event_id="unscored"),
        ],
    )

    found = await repository.find_anomalies_by_vehicle_and_time_range(
        "veh-tw-0142", FIXED_NOW - timedelta(days=1), FIXED_NOW + timedelta(days=1), 100
    )

    assert [stored.event_id for stored in found] == ["anomalous"]


async def test_events_ingested_today_produce_no_anomalies(repository) -> None:
    """Everything Phase 3 stores is PENDING, so this is correctly empty."""
    await _seed(repository, [stored_event(event_id=f"evt-{index}") for index in range(3)])

    found = await repository.find_anomalies_by_vehicle_and_time_range(
        "veh-tw-0142", FIXED_NOW - timedelta(days=1), FIXED_NOW + timedelta(days=1), 100
    )

    assert found == []
