"""The telemetry ingestion use case.

``IngestTelemetry`` is the single application-level entry point for the write
path. The REST router is a transport adapter in front of it, which is what lets
a future MQTT adapter reuse ingestion without duplicating any of it. No
transport type reaches this module; the transport only names itself, so the
stored event records where it came from.

The pipeline is built one phase at a time. Normalization and persistence run
for real. Inference does not exist yet, so an event is stored with a truthful
``PENDING`` verdict rather than a fabricated one.
"""

import logging
from dataclasses import dataclass

from app.application.errors import DuplicateEventIdError, PersistenceUnavailableError
from app.application.ports import TelemetryRepository
from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.inference import InferenceOutcome
from app.domain.normalizer import TelemetryNormalizer
from app.domain.stored import StoredTelemetryEvent
from app.domain.telemetry import SourceTelemetryEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    """What ingestion did with one event.

    Attributes:
        stored: The event now held under this ``event_id``. On a duplicate this
            is the event that was stored *first*, returned unchanged.
        duplicate: True when this ``event_id`` was already stored.
        conflict: True when the same ``event_id`` arrived carrying different
            content. The original is kept either way; this only records that it
            happened.
    """

    stored: StoredTelemetryEvent
    duplicate: bool
    conflict: bool = False


class IngestTelemetry:
    """Ingest one telemetry event."""

    def __init__(
        self, normalizer: TelemetryNormalizer, repository: TelemetryRepository
    ) -> None:
        self._normalizer = normalizer
        self._repository = repository

    async def execute(
        self,
        event: SourceTelemetryEvent,
        *,
        transport: str = "rest",
        api_version: str = "v1",
    ) -> IngestOutcome:
        """Normalize one event and store it exactly once.

        Args:
            event: The event as the device reported it, already validated
                against the telemetry contract but not yet normalized.
            transport: Which adapter accepted it, recorded as provenance.
            api_version: The contract version that adapter serves.

        Returns:
            The outcome, carrying the event now held under this ``event_id``.

        Raises:
            NormalizationError: If the event cannot be normalized.
            PersistenceUnavailableError: If the event could not be stored.
                Ingestion is fail-closed on this, so the caller must not treat
                the event as accepted.
        """
        canonical = self._normalizer.normalize(event)

        # An optimization, not the guarantee: it saves a doomed insert on an
        # obvious retry, but two concurrent requests can both read "absent".
        existing = await self._repository.find_by_event_id(canonical.event_id)
        if existing is not None:
            return self._resolve_duplicate(canonical, existing)

        stored = StoredTelemetryEvent(
            event=canonical,
            inference=InferenceOutcome.pending(),
            transport=transport,
            api_version=api_version,
        )
        try:
            await self._repository.save(stored)
        except DuplicateEventIdError:
            # The unique index closed the race the lookup above could not.
            existing = await self._repository.find_by_event_id(canonical.event_id)
            if existing is None:
                raise PersistenceUnavailableError(
                    "event_id reported as duplicate but no stored event could be read back"
                ) from None
            return self._resolve_duplicate(canonical, existing)

        return IngestOutcome(stored=stored, duplicate=False)

    def _resolve_duplicate(
        self, canonical: CanonicalTelemetryEvent, existing: StoredTelemetryEvent
    ) -> IngestOutcome:
        """Decide whether a duplicate ``event_id`` is a retry or a conflict.

        Either way the stored event wins and is returned untouched: first write
        wins, and nothing is ever overwritten. A conflict is logged rather than
        rejected, which is the approved behaviour until there is evidence that
        conflicts actually occur.
        """
        if existing.event.is_same_logical_event(canonical):
            return IngestOutcome(stored=existing, duplicate=True)

        logger.warning(
            "event_id reused with different content; keeping the stored event",
            extra={
                "event_id": canonical.event_id,
                "stored_vehicle_id": existing.event.vehicle_id,
                "incoming_vehicle_id": canonical.vehicle_id,
                "stored_event_time": existing.event.event_time.isoformat(),
                "incoming_event_time": canonical.event_time.isoformat(),
            },
        )
        return IngestOutcome(stored=existing, duplicate=True, conflict=True)
