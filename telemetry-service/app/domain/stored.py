"""A canonical telemetry event as it exists in the event store.

``StoredTelemetryEvent`` is the canonical event plus the state that only exists
once an event has been written: its inference verdict and how it arrived. It
holds no storage identity — MongoDB's ``_id`` never reaches this model.
"""

from dataclasses import dataclass

from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.inference import InferenceOutcome


@dataclass(frozen=True, slots=True)
class StoredTelemetryEvent:
    """One stored event: canonical measurements, verdict, and provenance."""

    event: CanonicalTelemetryEvent
    inference: InferenceOutcome
    transport: str
    api_version: str

    @property
    def event_id(self) -> str:
        """The domain identity, which is also the idempotency key."""
        return self.event.event_id
