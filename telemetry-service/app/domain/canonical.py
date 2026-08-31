"""Domain representation of telemetry after normalization.

``CanonicalTelemetryEvent`` is what exists *after* ``TelemetryNormalizer`` has
run: every metric is expressed in its canonical unit, and both timestamps are
timezone-aware UTC. Nothing upstream of the normalizer may be called canonical.

The model deliberately holds no persistence or inference concepts — no storage
identity, no write timestamps, no inference result. Those belong to the phases
that introduce them.

The domain depends on nothing outside the standard library: no FastAPI, no
Pydantic, no database driver, no ML library.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Mapping

from app.domain.units import MetricName


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be normalized to UTC, got offset {value.utcoffset()}")


@dataclass(frozen=True, slots=True)
class CanonicalTelemetryEvent:
    """One telemetry event in canonical units and UTC.

    Attributes:
        schema_version: Payload schema version the device emitted, carried
            through unchanged.
        event_id: The domain event identifier and idempotency key, opaque and
            unchanged. Not a storage key.
        vehicle_id: Opaque vehicle identifier, unchanged.
        site_id: Opaque site identifier, unchanged.
        event_time: When the device says the measurement happened, converted to
            UTC. The instant is preserved; only the offset representation
            changes.
        received_at: When this service accepted the event, stamped from the
            server clock in UTC. Never supplied by the client, and never equal
            to ``event_time`` by construction.
        metrics: Canonical value per metric, in ``CANONICAL_UNITS``. Values are
            bare numbers: the unit is a property of the schema, not of each
            value.
        source_units: The unit each metric arrived in, kept as provenance so a
            later reader can answer "what did the device actually send?". Never
            used for querying or scoring.
    """

    schema_version: str
    event_id: str
    vehicle_id: str
    site_id: str
    event_time: datetime
    received_at: datetime
    metrics: Mapping[MetricName, float]
    source_units: Mapping[MetricName, str]

    def __post_init__(self) -> None:
        _require_utc("event_time", self.event_time)
        _require_utc("received_at", self.received_at)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "source_units", MappingProxyType(dict(self.source_units)))

    @property
    def ingest_delay(self) -> timedelta:
        """How long the event took to arrive: ``received_at - event_time``.

        Positive for the normal case of a delayed event; negative only within
        the tolerated future clock skew.
        """
        return self.received_at - self.event_time
