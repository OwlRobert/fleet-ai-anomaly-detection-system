"""The telemetry ingestion use case.

``IngestTelemetry`` is the single application-level entry point for the write
path. The REST router is a transport adapter in front of it, which is what lets
a future MQTT adapter reuse ingestion without duplicating any of it. No
transport type reaches this module.

Phase 1 establishes the boundary and nothing else: the downstream pipeline
(``TelemetryNormalizer`` -> ``InferencePort`` -> ``TelemetryRepository``) does
not exist yet, so a validated event is explicitly refused rather than silently
accepted.
"""

from app.application.errors import CapabilityNotImplementedError
from app.domain.telemetry import SourceTelemetryEvent


class IngestTelemetry:
    """Ingest one telemetry event.

    Later phases give this use case its collaborators through ``__init__``;
    in Phase 1 it has none, because faking them would fake the pipeline.
    """

    def execute(self, event: SourceTelemetryEvent) -> None:
        """Accept a validated source telemetry event for ingestion.

        Args:
            event: The event as the device reported it, already validated
                against the telemetry contract but not yet normalized.

        Raises:
            CapabilityNotImplementedError: Always, in Phase 1. Normalization,
                inference and persistence are not implemented.
        """
        raise CapabilityNotImplementedError(
            capability="Telemetry ingestion",
            arrives_in="normalization, inference and persistence arrive in later phases",
        )
