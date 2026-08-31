"""How the MongoDB repository translates driver failures.

A stub collection stands in for the driver so the two failure paths — duplicate
key and everything else — can be exercised without a database.
"""

from datetime import timedelta

import pytest
from pymongo.errors import (
    AutoReconnect,
    DuplicateKeyError,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from app.application.errors import DuplicateEventIdError, PersistenceUnavailableError
from app.infrastructure.telemetry_repository import MongoTelemetryRepository
from tests.factories import FIXED_NOW, stored_event

pytestmark = pytest.mark.anyio


class FailingCollection:
    """A collection whose every operation raises the given driver error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def insert_one(self, document):
        raise self.error

    async def find_one(self, query):
        raise self.error

    def find(self, query):
        raise self.error


DRIVER_FAILURES = [
    pytest.param(ServerSelectionTimeoutError("no servers"), id="server-selection-timeout"),
    pytest.param(AutoReconnect("connection lost"), id="auto-reconnect"),
    pytest.param(OperationFailure("write failed"), id="operation-failure"),
]


@pytest.mark.parametrize("error", DRIVER_FAILURES)
async def test_a_driver_failure_on_write_becomes_persistence_unavailable(error: Exception) -> None:
    repository = MongoTelemetryRepository(FailingCollection(error))

    with pytest.raises(PersistenceUnavailableError):
        await repository.save(stored_event())


@pytest.mark.parametrize("error", DRIVER_FAILURES)
async def test_a_driver_failure_on_read_becomes_persistence_unavailable(error: Exception) -> None:
    repository = MongoTelemetryRepository(FailingCollection(error))

    with pytest.raises(PersistenceUnavailableError):
        await repository.find_by_event_id("evt-1")


async def test_a_driver_failure_on_a_range_query_becomes_persistence_unavailable() -> None:
    repository = MongoTelemetryRepository(FailingCollection(AutoReconnect("gone")))

    with pytest.raises(PersistenceUnavailableError):
        await repository.find_by_vehicle_and_time_range(
            "veh-tw-0142", FIXED_NOW - timedelta(days=1), FIXED_NOW, 100
        )


async def test_a_duplicate_key_error_is_not_a_persistence_failure() -> None:
    """It means the unique index did its job, not that the store is broken."""
    repository = MongoTelemetryRepository(FailingCollection(DuplicateKeyError("event_id")))

    with pytest.raises(DuplicateEventIdError) as raised:
        await repository.save(stored_event(event_id="evt-1"))

    assert raised.value.event_id == "evt-1"


async def test_a_duplicate_key_error_does_not_map_to_503() -> None:
    repository = MongoTelemetryRepository(FailingCollection(DuplicateKeyError("event_id")))

    with pytest.raises(DuplicateEventIdError):
        await repository.save(stored_event())


async def test_driver_exception_details_do_not_escape_the_repository() -> None:
    """PyMongo type names and messages stay inside the infrastructure layer."""
    repository = MongoTelemetryRepository(
        FailingCollection(ServerSelectionTimeoutError("mongodb://secret-host:27017 unreachable"))
    )

    with pytest.raises(PersistenceUnavailableError) as raised:
        await repository.save(stored_event())

    assert "secret-host" not in str(raised.value)
    assert "ServerSelectionTimeout" not in str(raised.value)
