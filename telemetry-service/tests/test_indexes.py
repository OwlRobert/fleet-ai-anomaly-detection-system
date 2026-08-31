"""The index specifications this service creates."""

import pytest
from pymongo import ASCENDING, DESCENDING

from app.infrastructure.mongo import (
    ANOMALY_INDEX,
    EVENT_ID_UNIQUE_INDEX,
    INDEX_SPECIFICATIONS,
    VEHICLE_TIME_INDEX,
    ensure_indexes,
)

pytestmark = pytest.mark.anyio

BY_NAME = {specification["name"]: specification for specification in INDEX_SPECIFICATIONS}


def test_only_the_approved_indexes_are_created() -> None:
    """Nothing speculative: every index serves a documented access pattern."""
    assert set(BY_NAME) == {
        EVENT_ID_UNIQUE_INDEX,
        VEHICLE_TIME_INDEX,
        ANOMALY_INDEX,
    }


def test_event_id_is_unique() -> None:
    """This constraint, not a prior lookup, is what makes ingestion idempotent."""
    specification = BY_NAME[EVENT_ID_UNIQUE_INDEX]

    assert specification["keys"] == [("event_id", ASCENDING)]
    assert specification["unique"] is True


def test_no_index_is_declared_on_the_storage_identity() -> None:
    for specification in INDEX_SPECIFICATIONS:
        assert all(field != "_id" for field, _ in specification["keys"])


def test_vehicle_history_is_indexed_newest_first() -> None:
    assert BY_NAME[VEHICLE_TIME_INDEX]["keys"] == [
        ("vehicle_id", ASCENDING),
        ("event_time", DESCENDING),
    ]


def test_site_id_is_stored_but_not_indexed() -> None:
    """It is on every event; nothing queries by it yet, so nothing indexes it."""
    indexed_fields = {field for spec in INDEX_SPECIFICATIONS for field, _ in spec["keys"]}

    assert "site_id" not in indexed_fields


def test_the_anomaly_index_is_partial_and_excludes_unscored_events() -> None:
    """PENDING is not 'not an anomaly', so it must not be in the anomaly index."""
    specification = BY_NAME[ANOMALY_INDEX]

    assert specification["keys"] == [("vehicle_id", ASCENDING), ("event_time", DESCENDING)]
    assert specification["partialFilterExpression"] == {
        "inference.is_anomaly": True,
        "inference.status": "COMPLETED",
    }


class RecordingCollection:
    """Captures the create_index calls ensure_indexes makes."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def create_index(self, keys, **options):
        self.calls.append((keys, options))
        return options["name"]


async def test_ensure_indexes_creates_every_specification() -> None:
    collection = RecordingCollection()

    created = await ensure_indexes(collection)

    assert created == [specification["name"] for specification in INDEX_SPECIFICATIONS]
    assert len(collection.calls) == len(INDEX_SPECIFICATIONS)


async def test_ensure_indexes_passes_uniqueness_and_partial_filters_through() -> None:
    collection = RecordingCollection()

    await ensure_indexes(collection)

    options_by_name = {options["name"]: options for _, options in collection.calls}
    assert options_by_name[EVENT_ID_UNIQUE_INDEX]["unique"] is True
    assert "partialFilterExpression" in options_by_name[ANOMALY_INDEX]
    assert "unique" not in options_by_name[VEHICLE_TIME_INDEX]


async def test_ensure_indexes_is_safe_to_run_repeatedly() -> None:
    """create_index is idempotent for an identical specification."""
    collection = RecordingCollection()

    await ensure_indexes(collection)
    await ensure_indexes(collection)

    assert len(collection.calls) == 2 * len(INDEX_SPECIFICATIONS)
