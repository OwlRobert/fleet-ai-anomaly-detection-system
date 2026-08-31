"""The telemetry ingestion use case.

``IngestTelemetry`` is the single application-level entry point for the write
path. The REST router is a transport adapter in front of it, which is what lets
a future MQTT adapter reuse ingestion without duplicating any of it. No
transport type reaches this module.

The pipeline is built one phase at a time. Normalization now runs for real; the
inference call and the persistence write that follow it do not exist yet, so a
normalized event is explicitly refused rather than silently accepted.
"""

from app.application.errors import CapabilityNotImplementedError
from app.domain.normalizer import TelemetryNormalizer
from app.domain.telemetry import SourceTelemetryEvent


class IngestTelemetry:
    """Ingest one telemetry event.

    Collaborators arrive with the phases that need them. It holds the
    normalizer today; ``InferencePort`` and ``TelemetryRepository`` join it
    later, and the routes do not change when they do.
    """

    def __init__(self, normalizer: TelemetryNormalizer) -> None:
        self._normalizer = normalizer

    def execute(self, event: SourceTelemetryEvent) -> None:
        """Accept a validated source telemetry event for ingestion.

        Normalization runs first and its rejections propagate to the caller. If
        it succeeds, the canonical event has nowhere to go yet: refusing here is
        the honest outcome, and returning a fabricated result would not be.

        Args:
            event: The event as the device reported it, already validated
                against the telemetry contract but not yet normalized.

        Raises:
            NormalizationError: If the event cannot be normalized, for instance
                because its clock skew exceeds the configured tolerance.
            CapabilityNotImplementedError: After successful normalization,
                because inference and persistence are not implemented.
        """
        self._normalizer.normalize(event)

        raise CapabilityNotImplementedError(
            capability="Telemetry ingestion",
            arrives_in="the event was normalized, but inference and persistence arrive in later phases",
        )
