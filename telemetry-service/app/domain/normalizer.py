"""Turn a validated source event into a canonical one.

The normalizer is the single place where source units become canonical units
and where a source offset becomes UTC. It is pure domain logic: it takes a
``SourceTelemetryEvent``, returns a new ``CanonicalTelemetryEvent``, and never
mutates its input. It knows nothing about HTTP, MQTT, files or any other
transport, which is what lets every transport share it.

Its only non-pure input is the server clock, injected as a plain callable so
tests can pin it. That is a testability seam, not a port.
"""

from datetime import datetime, timedelta, timezone
from typing import Callable

from app.domain.canonical import CanonicalTelemetryEvent
from app.domain.conversions import to_canonical
from app.domain.errors import ClockSkewFutureError, EventTooOldError
from app.domain.telemetry import SourceTelemetryEvent

DEFAULT_MAX_FUTURE_SKEW = timedelta(seconds=300)
"""Approved tolerance for an event_time ahead of received_at."""

DEFAULT_MAX_EVENT_AGE = timedelta(days=30)
"""Approved ingestion window for an event_time behind received_at."""


def utc_now() -> datetime:
    """The server clock, always timezone-aware UTC."""
    return datetime.now(timezone.utc)


class TelemetryNormalizer:
    """Normalize source telemetry into the canonical internal representation.

    Responsibilities, in order:

    1. stamp ``received_at`` from the server clock;
    2. reject an ``event_time`` outside the accepted window around it;
    3. convert ``event_time`` to UTC, preserving the instant;
    4. convert every metric into its canonical unit;
    5. keep the source unit of each metric as provenance.

    Ordering is *not* one of its concerns. Normalization is stateless: an event
    whose ``event_time`` precedes one already processed is still perfectly
    valid, because arrival order and event-time order are independent.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
        max_future_skew: timedelta = DEFAULT_MAX_FUTURE_SKEW,
        max_event_age: timedelta = DEFAULT_MAX_EVENT_AGE,
    ) -> None:
        self._clock = clock
        self._max_future_skew = max_future_skew
        self._max_event_age = max_event_age

    def normalize(self, source_event: SourceTelemetryEvent) -> CanonicalTelemetryEvent:
        """Produce the canonical event for one source event.

        Args:
            source_event: A validated event in its source units and offset.
                It is read, never modified.

        Returns:
            A new canonical event in canonical units and UTC.

        Raises:
            ClockSkewFutureError: ``event_time`` is further ahead of
                ``received_at`` than the configured tolerance.
            EventTooOldError: ``event_time`` is further behind ``received_at``
                than the configured ingestion window.
        """
        received_at = self._clock()
        event_time = source_event.event_time.astimezone(timezone.utc)
        self._check_clock_skew(event_time, received_at)

        return CanonicalTelemetryEvent(
            schema_version=source_event.schema_version,
            event_id=source_event.event_id,
            vehicle_id=source_event.vehicle_id,
            site_id=source_event.site_id,
            event_time=event_time,
            received_at=received_at,
            metrics={
                metric: to_canonical(metric, measurement.value, measurement.unit)
                for metric, measurement in source_event.metrics.items()
            },
            source_units={
                metric: measurement.unit
                for metric, measurement in source_event.metrics.items()
            },
        )

    def _check_clock_skew(self, event_time: datetime, received_at: datetime) -> None:
        """Bound how far ``event_time`` may sit either side of ``received_at``.

        Both bounds are inclusive: an event exactly at the limit is accepted.
        Neither bound clamps or rewrites the timestamp — a rejected event stays
        rejected, so a broken device clock stays visible.
        """
        ahead = event_time - received_at
        if ahead > self._max_future_skew:
            raise ClockSkewFutureError(
                f"event_time is {ahead} ahead of received_at, "
                f"which exceeds the {self._max_future_skew} tolerance",
                event_time=event_time,
                received_at=received_at,
                limit=self._max_future_skew,
            )

        behind = received_at - event_time
        if behind > self._max_event_age:
            raise EventTooOldError(
                f"event_time is {behind} behind received_at, "
                f"which exceeds the {self._max_event_age} ingestion window",
                event_time=event_time,
                received_at=received_at,
                limit=self._max_event_age,
            )
