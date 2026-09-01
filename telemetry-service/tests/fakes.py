"""Test doubles at the repository port.

``InMemoryTelemetryRepository`` is a double for the *port*, not for MongoDB. It
enforces the one guarantee the application depends on — uniqueness of
``event_id`` — and applies the same ordering and range semantics, so use-case
behaviour can be tested without a database. It proves nothing about MongoDB
itself; that is what the integration tests are for.
"""

from datetime import datetime
from typing import Callable, Mapping

from app.application.errors import (
    DuplicateEventIdError,
    InferenceFailedError,
    PersistenceUnavailableError,
)
from app.domain.inference import InferenceErrorCode, InferenceOutcome
from app.domain.stored import StoredTelemetryEvent
from app.domain.units import MetricName


class InMemoryTelemetryRepository:
    """A minimal, faithful stand-in for the telemetry event store."""

    def __init__(self) -> None:
        self.events: list[StoredTelemetryEvent] = []
        self.save_calls = 0
        #: Set to raise on the next call, to simulate an unavailable store.
        self.fail_with: Exception | None = None
        #: Called before each save, to simulate a concurrent writer winning the race.
        self.before_save: Callable[[], None] | None = None

    def _guard(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    async def save(self, event: StoredTelemetryEvent) -> None:
        self._guard()
        if self.before_save is not None:
            self.before_save()
        self.save_calls += 1
        # The uniqueness check lives here, in the store, exactly as the unique
        # index does: it is what makes a lost race impossible to turn into a
        # second record.
        if any(stored.event_id == event.event_id for stored in self.events):
            raise DuplicateEventIdError(event.event_id)
        self.events.append(event)

    async def find_by_event_id(self, event_id: str) -> StoredTelemetryEvent | None:
        self._guard()
        return next((stored for stored in self.events if stored.event_id == event_id), None)

    async def find_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        self._guard()
        return self._ordered(
            stored
            for stored in self.events
            if stored.event.vehicle_id == vehicle_id and start <= stored.event.event_time < end
        )[:limit]

    async def find_anomalies_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        self._guard()
        return self._ordered(
            stored
            for stored in self.events
            if stored.event.vehicle_id == vehicle_id
            and start <= stored.event.event_time < end
            and stored.inference.is_confirmed_anomaly
        )[:limit]

    @staticmethod
    def _ordered(events) -> list[StoredTelemetryEvent]:
        """Newest ``event_time`` first, ties broken by ``event_id``."""
        return sorted(events, key=lambda s: (s.event.event_time, s.event_id), reverse=True)


class UnavailableTelemetryRepository:
    """A store that is simply down. Every call fails closed."""

    async def save(self, event: StoredTelemetryEvent) -> None:
        raise PersistenceUnavailableError("telemetry could not be stored")

    async def find_by_event_id(self, event_id: str) -> StoredTelemetryEvent | None:
        raise PersistenceUnavailableError("telemetry store could not be queried")

    async def find_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        raise PersistenceUnavailableError("telemetry store could not be queried")

    async def find_anomalies_by_vehicle_and_time_range(
        self, vehicle_id: str, start: datetime, end: datetime, limit: int
    ) -> list[StoredTelemetryEvent]:
        raise PersistenceUnavailableError("telemetry store could not be queried")


class StubInferencePort:
    """A stand-in for the Inference Service at the port boundary.

    It records what it was asked to score, so tests can prove the *canonical*
    values crossed the boundary, and can be told to fail with any approved
    error code.
    """

    def __init__(
        self,
        *,
        is_anomaly: bool = False,
        anomaly_score: float = -0.1034,
        model_name: str = "isolation-forest-telemetry",
        model_version: str = "0.1.0",
        fail_with: InferenceErrorCode | None = None,
    ) -> None:
        self.calls: list[Mapping[MetricName, float]] = []
        self.fail_with = fail_with
        self._outcome = InferenceOutcome.completed(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            model_name=model_name,
            model_version=model_version,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def predict(self, features: Mapping[MetricName, float]) -> InferenceOutcome:
        self.calls.append(dict(features))
        if self.fail_with is not None:
            raise InferenceFailedError(self.fail_with, "stubbed failure")
        return self._outcome
