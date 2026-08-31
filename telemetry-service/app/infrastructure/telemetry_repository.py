"""MongoDB implementation of ``TelemetryRepository``.

Everything MongoDB-specific stops here: driver exceptions become application
errors, documents become domain objects, and ``_id`` is never read.
"""

from datetime import datetime

from pymongo import DESCENDING
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.application.errors import DuplicateEventIdError, PersistenceUnavailableError
from app.domain.inference import InferenceStatus
from app.domain.stored import StoredTelemetryEvent
from app.infrastructure.documents import from_document, to_document


class MongoTelemetryRepository:
    """Stores telemetry events in one MongoDB collection."""

    def __init__(self, collection: AsyncCollection) -> None:
        self._collection = collection

    async def save(self, event: StoredTelemetryEvent) -> None:
        """Insert one event, letting the unique index reject a duplicate.

        The uniqueness of ``event_id`` is decided by the database, so two
        concurrent requests that both saw "not stored" cannot both succeed.
        """
        try:
            await self._collection.insert_one(to_document(event))
        except DuplicateKeyError as exc:
            raise DuplicateEventIdError(event.event_id) from exc
        except PyMongoError as exc:
            raise PersistenceUnavailableError("telemetry could not be stored") from exc

    async def find_by_event_id(self, event_id: str) -> StoredTelemetryEvent | None:
        try:
            document = await self._collection.find_one({"event_id": event_id})
        except PyMongoError as exc:
            raise PersistenceUnavailableError("telemetry store could not be queried") from exc
        return from_document(document) if document is not None else None

    async def find_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        """Events in ``[start, end)`` — start inclusive, end exclusive."""
        return await self._find(
            {"vehicle_id": vehicle_id, "event_time": {"$gte": start, "$lt": end}}, limit
        )

    async def find_anomalies_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        """As above, restricted to events a completed run scored as anomalous.

        The status condition is not redundant: an unscored event has no verdict,
        and must never be returned merely because a missing field could be read
        as false.
        """
        return await self._find(
            {
                "vehicle_id": vehicle_id,
                "event_time": {"$gte": start, "$lt": end},
                "inference.status": InferenceStatus.COMPLETED.value,
                "inference.is_anomaly": True,
            },
            limit,
        )

    async def _find(self, query: dict, limit: int) -> list[StoredTelemetryEvent]:
        """Run a bounded query ordered by ``event_time`` descending.

        Ordering is by ``event_time``, never by insertion order or ``_id``, so a
        late-arriving event appears in its correct temporal position. ``event_id``
        breaks ties into a stable total order. The limit is always applied here,
        so no caller can trigger an unbounded read.
        """
        try:
            cursor = (
                self._collection.find(query)
                .sort([("event_time", DESCENDING), ("event_id", DESCENDING)])
                .limit(limit)
            )
            return [from_document(document) async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceUnavailableError("telemetry store could not be queried") from exc
