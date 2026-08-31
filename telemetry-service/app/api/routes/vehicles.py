"""Per-vehicle history queries.

Reads are thin query handlers rather than application use cases, matching the
approved design: the service has one application use case, and it is for the
write path. There is no repository yet, so neither endpoint can return events —
and neither invents any.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.errors import ErrorEnvelope
from app.api.schemas import TimeRangeQuery, VehicleTelemetryPage
from app.application.errors import CapabilityNotImplementedError

router = APIRouter(prefix="/api/v1/vehicles", tags=["vehicles"])

_QUERY_RESPONSES = {
    200: {"model": VehicleTelemetryPage, "description": "A page of events (not yet implemented)."},
    422: {"model": ErrorEnvelope, "description": "Invalid time range or page size."},
    501: {"model": ErrorEnvelope, "description": "Telemetry storage is not implemented in this phase."},
}


@router.get(
    "/{vehicle_id}/telemetry",
    summary="Telemetry history for one vehicle",
    description=(
        "Returns the vehicle's events within `[start, end)`, ordered by "
        "`event_time` descending. Both bounds must be timezone-aware.\n\n"
        "Query parameters are validated, but there is no telemetry store yet, so "
        "the endpoint answers **501 Not Implemented** instead of an empty or "
        "fabricated page."
    ),
    responses=_QUERY_RESPONSES,
)
def get_vehicle_telemetry(
    vehicle_id: str,
    time_range: Annotated[TimeRangeQuery, Query()],
) -> None:
    raise CapabilityNotImplementedError(
        capability="Telemetry history queries",
        arrives_in="the telemetry store arrives in a later phase",
    )


@router.get(
    "/{vehicle_id}/anomalies",
    summary="Anomalous telemetry events for one vehicle",
    description=(
        "Same parameters and ordering as the telemetry history, restricted to "
        "events the model scored as anomalous. Events whose inference did not "
        "complete are never returned here: an unscored event is not a "
        "non-anomaly.\n\n"
        "Query parameters are validated, but there is no telemetry store and no "
        "inference yet, so the endpoint answers **501 Not Implemented**."
    ),
    responses=_QUERY_RESPONSES,
)
def get_vehicle_anomalies(
    vehicle_id: str,
    time_range: Annotated[TimeRangeQuery, Query()],
) -> None:
    raise CapabilityNotImplementedError(
        capability="Anomaly history queries",
        arrives_in="the telemetry store and inference arrive in later phases",
    )
