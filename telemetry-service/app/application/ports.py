"""Application ports.

One port, introduced now because Phase 3 finally has two things to put behind
it: a MongoDB implementation and an in-memory fake for tests. It is a specific
interface shaped by the four things the application actually does, not a
generic repository.

The port speaks **domain identity only**. ``event_id`` crosses it; MongoDB's
``_id`` never does, which is what lets an alternate store reproduce the same
idempotency semantics without leaking its own identity model upward.
"""

from datetime import datetime
from typing import Protocol

from app.domain.stored import StoredTelemetryEvent


class TelemetryRepository(Protocol):
    """The telemetry event store, as the application needs it."""

    async def save(self, event: StoredTelemetryEvent) -> None:
        """Store one event that has never been stored before.

        Raises:
            DuplicateEventIdError: If an event with the same ``event_id``
                already exists. Uniqueness is enforced by the store, not by a
                prior lookup, so this is the authoritative duplicate signal.
            PersistenceUnavailableError: If the store could not be reached or
                the write could not be completed.
        """
        ...

    async def find_by_event_id(self, event_id: str) -> StoredTelemetryEvent | None:
        """Return the stored event with this ``event_id``, if there is one."""
        ...

    async def find_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        """Events for one vehicle in ``[start, end)``, newest ``event_time`` first."""
        ...

    async def find_anomalies_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        """As above, restricted to events a completed run scored as anomalous."""
        ...
