"""Wiring for the API layer.

FastAPI's own ``Depends`` is the only injection mechanism; no container and no
DI framework. Later phases construct ``IngestTelemetry`` with its collaborators
here, and nothing in the routes changes.
"""

from app.application.ingest_telemetry import IngestTelemetry


def get_ingest_telemetry() -> IngestTelemetry:
    """Provide the telemetry ingestion use case."""
    return IngestTelemetry()
