"""MongoDB client lifecycle and index management.

One client per process, created at application startup and closed at shutdown.
A client per request would rebuild the connection pool on every call.

The driver is ``pymongo``'s official async API (``AsyncMongoClient``, available
since PyMongo 4.13). Motor, the previous third-party async driver, is
deprecated, so this is the supported path.
"""

from typing import Any

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

EVENT_ID_UNIQUE_INDEX = "uniq_event_id"
VEHICLE_TIME_INDEX = "vehicle_id_event_time_desc"
ANOMALY_INDEX = "vehicle_id_event_time_desc_anomalies"

INDEX_SPECIFICATIONS: tuple[dict[str, Any], ...] = (
    {
        # Idempotency. This unique constraint - not a prior lookup - is what
        # makes a concurrent retry incapable of creating a second record.
        "name": EVENT_ID_UNIQUE_INDEX,
        "keys": [("event_id", ASCENDING)],
        "unique": True,
    },
    {
        # GET /api/v1/vehicles/{id}/telemetry, newest event_time first.
        "name": VEHICLE_TIME_INDEX,
        "keys": [("vehicle_id", ASCENDING), ("event_time", DESCENDING)],
    },
    {
        # GET /api/v1/vehicles/{id}/anomalies. Partial, so the index stays
        # proportional to the number of anomalies rather than to every event.
        # The filter also excludes unscored events: PENDING is not "not an
        # anomaly", it is "no verdict".
        "name": ANOMALY_INDEX,
        "keys": [("vehicle_id", ASCENDING), ("event_time", DESCENDING)],
        "partialFilterExpression": {
            "inference.is_anomaly": True,
            "inference.status": "COMPLETED",
        },
    },
)
"""Every index this service creates.

Each one serves an access pattern the service actually has today. ``site_id`` is
stored on every event but is deliberately not indexed: no query filters on it
yet, and an index without a caller is speculation, not preparation."""


def create_client(uri: str, timeout_ms: int) -> AsyncMongoClient:
    """Build the process-wide client.

    ``tz_aware=True`` matters: without it the driver returns naive datetimes and
    UTC-correct writes would come back as naive values, quietly reintroducing
    the ambiguity normalization exists to remove.
    """
    return AsyncMongoClient(
        uri,
        tz_aware=True,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )


async def ensure_indexes(collection: AsyncCollection) -> list[str]:
    """Create the required indexes if they are not already there.

    ``create_index`` is idempotent for an identical specification, so this is
    safe to run on every startup.
    """
    created: list[str] = []
    for specification in INDEX_SPECIFICATIONS:
        options = {
            key: value for key, value in specification.items() if key not in {"keys", "name"}
        }
        created.append(
            await collection.create_index(
                specification["keys"], name=specification["name"], **options
            )
        )
    return created


def get_collection(database: AsyncDatabase, name: str) -> AsyncCollection:
    """The telemetry event collection."""
    return database[name]
