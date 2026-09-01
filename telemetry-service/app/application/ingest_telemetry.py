"""The telemetry ingestion use case.

``IngestTelemetry`` is the single application-level entry point for the write
path. The REST router is a transport adapter in front of it, which is what lets
a future MQTT adapter reuse ingestion without duplicating any of it. No
transport type reaches this module; the transport only names itself, so the
stored event records where it came from.

The write path is now complete and synchronous: normalize, score, store. The
two downstream dependencies have deliberately opposite failure policies —
inference failure is fail-open (store the telemetry, record that no verdict was
obtained), persistence failure is fail-closed (tell the client nothing was
accepted).
"""

import logging
from dataclasses import dataclass

from app.application.errors import (
    DuplicateEventIdError,
    InferenceFailedError,
    PersistenceUnavailableError,
)
from app.application.ports import InferencePort, TelemetryRepository
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
        self,
        normalizer: TelemetryNormalizer,
        inference: InferencePort,
        repository: TelemetryRepository,
    ) -> None:
        self._normalizer = normalizer
        self._inference = inference
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
                the event as accepted. Inference failure, by contrast, is
                fail-open and never raises out of here.
        """
        canonical = self._normalizer.normalize(event)

        # An optimization, not the guarantee: it saves a doomed insert on an
        # obvious retry, but two concurrent requests can both read "absent".
        # It also means a retry never re-scores: the stored verdict is returned
        # exactly as it was first written, whether COMPLETED or FAILED.
        existing = await self._repository.find_by_event_id(canonical.event_id)
        if existing is not None:
            return self._resolve_duplicate(canonical, existing)

        stored = StoredTelemetryEvent(
            event=canonical,
            inference=await self._score(canonical),
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

    async def _score(self, canonical: CanonicalTelemetryEvent) -> InferenceOutcome:
        """Ask the model for a verdict, and carry on without one if it cannot.

        This is the fail-open boundary. A failure here degrades the *result*,
        never the measurement: the event is still stored, carrying the reason no
        verdict exists.
        """
        try:
            return await self._inference.predict(canonical.metrics)
        except InferenceFailedError as exc:
            logger.warning(
                "storing telemetry without a verdict; inference did not complete",
                extra={
                    "event_id": canonical.event_id,
                    "vehicle_id": canonical.vehicle_id,
                    "site_id": canonical.site_id,
                    "error_code": exc.error_code.value,
                },
            )
            return InferenceOutcome.failed(exc.error_code)

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
