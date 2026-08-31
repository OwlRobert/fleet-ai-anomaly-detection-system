"""Domain representation of telemetry as the device reported it.

``SourceTelemetryEvent`` is the *validated but not yet normalized* event: values
are still expressed in their source units and ``event_time`` still carries the
offset the device sent. Converting to canonical units and UTC produces a
separate canonical representation, which is Phase 2 work and is deliberately
absent here.

The domain depends on nothing outside the standard library: no FastAPI, no
Pydantic, no database driver, no ML library.
"""

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from app.domain.units import MetricName


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured value together with the unit the device reported it in."""

    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class SourceTelemetryEvent:
    """A single telemetry event in its source units and source offset.

    ``event_id`` is the domain-level event identifier and the future idempotency
    key. It is opaque to this service and carries no storage-engine meaning.
    """

    schema_version: str
    event_id: str
    vehicle_id: str
    site_id: str
    event_time: datetime
    metrics: Mapping[MetricName, Measurement]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
