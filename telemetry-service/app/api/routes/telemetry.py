"""Telemetry ingestion — the REST transport adapter.

The router validates the payload against the telemetry contract, translates it
into a domain event and hands it to the ``IngestTelemetry`` use case. It holds
no orchestration of its own, which is what allows a future MQTT adapter to
invoke the same use case.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_ingest_telemetry
from app.api.errors import ErrorEnvelope
from app.api.schemas import TelemetryEventResponse, TelemetryIngestRequest
from app.application.ingest_telemetry import IngestTelemetry

router = APIRouter(prefix="/api/v1", tags=["telemetry"])

API_VERSION = "v1"
TRANSPORT = "rest"


@router.post(
    "/telemetry",
    status_code=status.HTTP_201_CREATED,
    response_model=TelemetryEventResponse,
    summary="Ingest one telemetry event",
    description=(
        "Accepts a single telemetry event in its **source units** — every metric "
        "carries its own explicit unit, and there is no request-level unit "
        "system. `event_time` must be timezone-aware.\n\n"
        "The payload is validated, normalized to canonical units and UTC, "
        "stamped with `received_at`, and stored exactly once.\n\n"
        "Ingestion is idempotent on `event_id`: a first event answers **201 "
        "Created** with `duplicate: false`, and any retry answers **200 OK** "
        "with `duplicate: true` and the event that was stored first, unchanged. "
        "Uniqueness is enforced by the store, so concurrent retries cannot both "
        "create a record.\n\n"
        "Inference is not implemented yet, so a newly stored event carries "
        "`inference.status: \"PENDING\"` — stored but never scored. No anomaly "
        "verdict is invented, and a pending event is never returned by the "
        "anomalies endpoint."
    ),
    responses={
        200: {"model": TelemetryEventResponse, "description": "Duplicate event_id; the event stored first is returned unchanged."},
        201: {"model": TelemetryEventResponse, "description": "Event stored. Not yet scored."},
        422: {"model": ErrorEnvelope, "description": "Payload failed contract validation or was rejected by clock-skew bounds."},
        503: {"model": ErrorEnvelope, "description": "Telemetry could not be stored; the event was not accepted. Retry is safe."},
    },
)
async def ingest_telemetry(
    request: TelemetryIngestRequest,
    response: Response,
    use_case: Annotated[IngestTelemetry, Depends(get_ingest_telemetry)],
) -> TelemetryEventResponse:
    outcome = await use_case.execute(
        request.to_domain_event(), transport=TRANSPORT, api_version=API_VERSION
    )
    if outcome.duplicate:
        response.status_code = status.HTTP_200_OK
    return TelemetryEventResponse.from_stored(outcome.stored, duplicate=outcome.duplicate)
