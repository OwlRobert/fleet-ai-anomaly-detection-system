"""Wiring for the API layer.

FastAPI's own ``Depends`` is the only injection mechanism; no container and no
DI framework. This is where configuration becomes constructor arguments, so the
domain keeps taking plain values and the routes keep taking a finished use case.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import Depends

from app.application.ingest_telemetry import IngestTelemetry
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


def get_ingest_telemetry(
    normalizer: Annotated[TelemetryNormalizer, Depends(get_telemetry_normalizer)],
) -> IngestTelemetry:
    """Provide the telemetry ingestion use case."""
    return IngestTelemetry(normalizer=normalizer)
