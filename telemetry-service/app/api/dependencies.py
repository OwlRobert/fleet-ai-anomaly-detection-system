"""Wiring for the API layer.

FastAPI's own ``Depends`` is the only injection mechanism; no container and no
DI framework. This is where configuration and the process-wide MongoDB client
become constructor arguments, so the domain keeps taking plain values and the
routes keep taking finished collaborators.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import Depends, Request

from app.application.errors import PersistenceUnavailableError
from app.application.ingest_telemetry import IngestTelemetry
from app.application.ports import TelemetryRepository
from app.core.config import Settings, get_settings
from app.domain.normalizer import TelemetryNormalizer


def get_telemetry_normalizer(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TelemetryNormalizer:
    """Provide the normalizer, with the approved clock-skew bounds applied."""
    return TelemetryNormalizer(
        max_future_skew=timedelta(seconds=settings.max_clock_skew_future_seconds),
        max_event_age=timedelta(days=settings.max_event_age_days),
    )


def get_telemetry_repository(request: Request) -> TelemetryRepository:
    """Provide the repository built once at application startup.

    One client and one repository per process, not per request.
    """
    repository = getattr(request.app.state, "telemetry_repository", None)
    if repository is None:
        raise PersistenceUnavailableError("telemetry store is not configured")
    return repository


def get_ingest_telemetry(
    normalizer: Annotated[TelemetryNormalizer, Depends(get_telemetry_normalizer)],
    repository: Annotated[TelemetryRepository, Depends(get_telemetry_repository)],
) -> IngestTelemetry:
    """Provide the telemetry ingestion use case."""
    return IngestTelemetry(normalizer=normalizer, repository=repository)
