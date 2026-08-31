"""Mapping between the domain and the MongoDB document."""

from datetime import timedelta, timezone

import pytest

from app.domain.inference import InferenceStatus
from app.domain.units import MetricName
from app.infrastructure.documents import from_document, to_document
from tests.factories import FIXED_NOW, scored, stored_event


def test_document_carries_every_canonical_field() -> None:
    document = to_document(stored_event())

    assert set(document) == {
        "event_id",
        "schema_version",
        "vehicle_id",
        "site_id",
        "event_time",
        "received_at",
        "metrics",
        "source_units",
        "inference",
        "ingest",
    }


def test_document_holds_no_storage_identity() -> None:
    """_id is MongoDB's to assign; the mapper never writes one."""
    assert "_id" not in to_document(stored_event())


def test_event_id_is_a_plain_field_not_the_primary_key() -> None:
    document = to_document(stored_event(event_id="evt-1"))

    assert document["event_id"] == "evt-1"
    assert "_id" not in document


def test_metrics_are_stored_as_bare_canonical_numbers() -> None:
    document = to_document(stored_event())

    assert document["metrics"]["speed"] == pytest.approx(51.9)
    assert isinstance(document["metrics"]["speed"], float)


def test_source_units_are_persisted_as_provenance() -> None:
    document = to_document(stored_event(source_units={metric: "mph" for metric in MetricName}))

    assert document["source_units"]["speed"] == "mph"


def test_unscored_events_persist_a_pending_inference_block() -> None:
    document = to_document(stored_event())

    assert document["inference"] == {
        "status": "PENDING",
        "is_anomaly": None,
        "anomaly_score": None,
        "model_name": None,
        "model_version": None,
        "error_code": None,
    }


def test_ingest_provenance_is_persisted() -> None:
    document = to_document(stored_event(transport="rest", api_version="v1"))

    assert document["ingest"] == {"transport": "rest", "api_version": "v1"}


def test_no_created_at_is_written() -> None:
    """The approved document shape has no created_at; received_at is the stamp."""
    assert "created_at" not in to_document(stored_event())


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_round_trip_preserves_the_event() -> None:
    original = stored_event(inference=scored(is_anomaly=True))

    restored = from_document(to_document(original))

    assert restored == original


def test_round_trip_preserves_the_utc_instant() -> None:
    event_time = FIXED_NOW - timedelta(hours=7, minutes=13)
    original = stored_event(event_time=event_time, received_at=FIXED_NOW)

    restored = from_document(to_document(original))

    assert restored.event.event_time == event_time
    assert restored.event.received_at == FIXED_NOW
    assert restored.event.event_time.utcoffset() == timedelta(0)


def test_naive_timestamps_from_the_driver_are_read_back_as_utc() -> None:
    """A client configured without tz_aware would hand back naive values."""
    document = to_document(stored_event())
    document["event_time"] = document["event_time"].replace(tzinfo=None)
    document["received_at"] = document["received_at"].replace(tzinfo=None)

    restored = from_document(document)

    assert restored.event.event_time.tzinfo is not None
    assert restored.event.event_time.utcoffset() == timedelta(0)


def test_offset_timestamps_from_the_driver_are_converted_not_truncated() -> None:
    document = to_document(stored_event())
    instant = document["event_time"]
    document["event_time"] = instant.astimezone(timezone(timedelta(hours=8)))

    restored = from_document(document)

    assert restored.event.event_time == instant


def test_storage_identity_is_ignored_when_reading() -> None:
    document = to_document(stored_event())
    document["_id"] = "68b1f0c2a3d4e5f6a7b8c9d0"

    restored = from_document(document)

    assert not hasattr(restored, "_id")
    assert restored.event_id == stored_event().event_id


def test_a_scored_document_reads_back_its_verdict() -> None:
    restored = from_document(to_document(stored_event(inference=scored(is_anomaly=True))))

    assert restored.inference.status is InferenceStatus.COMPLETED
    assert restored.inference.is_anomaly is True
    assert restored.inference.is_confirmed_anomaly is True
