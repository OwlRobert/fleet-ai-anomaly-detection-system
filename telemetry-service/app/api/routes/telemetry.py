"""Telemetry ingestion — the REST transport adapter.

The router validates the payload against the telemetry contract, translates it
into a domain event and hands it to the ``IngestTelemetry`` use case. It holds
no orchestration of its own, which is what allows a future MQTT adapter to
invoke the same use case.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_ingest_telemetry
from app.api.errors import ErrorEnvelope
from app.api.schemas import TelemetryEventResponse, TelemetryIngestRequest
from app.application.ingest_telemetry import IngestTelemetry

router = APIRouter(prefix="/api/v1", tags=["telemetry"])


@router.post(
    "/telemetry",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest one telemetry event",
    description=(
        "Accepts a single telemetry event in its **source units** — every metric "
        "carries its own explicit unit, and there is no request-level unit "
        "system. `event_time` must be timezone-aware.\n\n"
        "The payload is validated against the telemetry contract and then "
        "normalized: units are converted to canonical units, `event_time` is "
        "converted to UTC, and `received_at` is stamped from the server clock. "
        "An `event_time` outside the accepted window around `received_at` is "
        "rejected with `CLOCK_SKEW_FUTURE` or `EVENT_TOO_OLD`.\n\n"
        "Inference and persistence are not implemented yet, so a successfully "
        "normalized event is answered with **501 Not Implemented** rather than "
        "a fabricated result. The 201 schema below is the contract this "
        "endpoint will fulfil once those capabilities exist."
    ),
    responses={
        201: {"model": TelemetryEventResponse, "description": "Event stored and scored (not yet implemented)."},
        422: {"model": ErrorEnvelope, "description": "Payload failed contract validation or was rejected by clock-skew bounds."},
        501: {"model": ErrorEnvelope, "description": "Event normalized, but inference and persistence are not implemented yet."},
    },
)
def ingest_telemetry(
    request: TelemetryIngestRequest,
    use_case: Annotated[IngestTelemetry, Depends(get_ingest_telemetry)],
) -> None:
    # The use case owns the outcome: it normalizes, then refuses because there
    # is nowhere to put the result yet. Persistence turns that into a 201 body.
    use_case.execute(request.to_domain_event())
