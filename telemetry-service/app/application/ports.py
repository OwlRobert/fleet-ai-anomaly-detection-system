"""Application ports.

Two ports, each introduced only when a second implementation or an external
system with its own failure modes actually arrived. Both are specific
interfaces shaped by what the application does, not generic infrastructure.

``TelemetryRepository`` speaks **domain identity only**. ``event_id`` crosses
it; MongoDB's ``_id`` never does, which is what lets an alternate store
reproduce the same idempotency semantics without leaking its own identity model
upward.

``InferencePort`` speaks **canonical features only**. No HTTP type, no URL and
no client library reaches the application layer through it.
"""

from datetime import datetime
from typing import Mapping, Protocol

from app.domain.inference import InferenceOutcome
from app.domain.stored import StoredTelemetryEvent
from app.domain.units import MetricName


class InferencePort(Protocol):
    """The anomaly model, as the application needs it."""

    async def predict(self, features: Mapping[MetricName, float]) -> InferenceOutcome:
        """Score one canonical feature vector.

        Args:
            features: Canonical values keyed by metric, already normalized.
                Source units never reach this boundary.

        Returns:
            A ``COMPLETED`` outcome carrying the model's verdict and identity.

        Raises:
            InferenceFailedError: If no verdict could be obtained, carrying the
                error code describing the failure class. Ingestion is fail-open
                on this: the caller records the failure and stores the telemetry
                anyway.
        """
        ...


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
