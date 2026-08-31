"""Mapping between the domain and the stored MongoDB document.

All knowledge of the document shape lives here. The domain models know nothing
about BSON, collections or ``_id``; this module is the only place that does.

``_id`` is never read and never written: MongoDB assigns its own, and it stops
at this boundary. The domain identity is ``event_id``, an ordinary field under
a unique index.
"""

from datetime import datetime, timezone
from typing import Any, Mapping

from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.inference import InferenceOutcome, InferenceStatus
from app.domain.stored import StoredTelemetryEvent
from app.domain.units import MetricName


def to_document(stored: StoredTelemetryEvent) -> dict[str, Any]:
    """Render a stored event as the MongoDB document to insert.

    Datetimes are handed to the driver as timezone-aware UTC. BSON keeps them
    in UTC at millisecond resolution, and the client is configured ``tz_aware``
    so they come back aware rather than naive.
    """
    event = stored.event
    return {
        "event_id": event.event_id,
        "schema_version": event.schema_version,
        "vehicle_id": event.vehicle_id,
        "site_id": event.site_id,
        "event_time": event.event_time,
        "received_at": event.received_at,
        "metrics": {metric.value: value for metric, value in event.metrics.items()},
        "source_units": {metric.value: unit for metric, unit in event.source_units.items()},
        "inference": {
            "status": stored.inference.status.value,
            "is_anomaly": stored.inference.is_anomaly,
            "anomaly_score": stored.inference.anomaly_score,
            "model_name": stored.inference.model_name,
            "model_version": stored.inference.model_version,
            "error_code": stored.inference.error_code,
        },
        "ingest": {"transport": stored.transport, "api_version": stored.api_version},
    }


def _as_utc(value: datetime) -> datetime:
    """Guarantee an aware UTC datetime, whatever the driver handed back."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def from_document(document: Mapping[str, Any]) -> StoredTelemetryEvent:
    """Rebuild a stored event from its MongoDB document.

    ``_id`` is ignored: it is storage identity and has no meaning above this
    module.
    """
    inference = document.get("inference") or {}
    return StoredTelemetryEvent(
        event=CanonicalTelemetryEvent(
            schema_version=document["schema_version"],
            event_id=document["event_id"],
            vehicle_id=document["vehicle_id"],
            site_id=document["site_id"],
            event_time=_as_utc(document["event_time"]),
            received_at=_as_utc(document["received_at"]),
            metrics={MetricName(name): float(value) for name, value in document["metrics"].items()},
            source_units={
                MetricName(name): unit for name, unit in document["source_units"].items()
            },
        ),
        inference=InferenceOutcome(
            status=InferenceStatus(inference.get("status", InferenceStatus.PENDING.value)),
            is_anomaly=inference.get("is_anomaly"),
            anomaly_score=inference.get("anomaly_score"),
            model_name=inference.get("model_name"),
            model_version=inference.get("model_version"),
            error_code=inference.get("error_code"),
        ),
        transport=(document.get("ingest") or {}).get("transport", "unknown"),
        api_version=(document.get("ingest") or {}).get("api_version", "unknown"),
    )
