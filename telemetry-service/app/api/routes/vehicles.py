"""Per-vehicle history queries.

Reads are thin query handlers rather than application use cases, matching the
approved design: the service has one application use case, and it is for the
write path. The handlers translate query parameters and hand off to the
repository; no MongoDB type appears here.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_telemetry_repository
from app.api.errors import ErrorEnvelope
from app.api.schemas import TimeRangeQuery, VehicleTelemetryPage
from app.application.ports import TelemetryRepository

router = APIRouter(prefix="/api/v1/vehicles", tags=["vehicles"])

_QUERY_RESPONSES = {
    200: {"model": VehicleTelemetryPage, "description": "A page of events, newest event_time first."},
    422: {"model": ErrorEnvelope, "description": "Invalid time range or page size."},
    503: {"model": ErrorEnvelope, "description": "The telemetry store could not be queried."},
}


@router.get(
    "/{vehicle_id}/telemetry",
    response_model=VehicleTelemetryPage,
    summary="Telemetry history for one vehicle",
    description=(
        "Returns the vehicle's events within `[start, end)` — `start` inclusive, "
        "`end` exclusive — ordered by `event_time` **descending**. Both bounds "
        "must be timezone-aware.\n\n"
        "Ordering is by `event_time`, never by insertion order, so an event that "
        "arrived late still appears in its correct temporal position.\n\n"
        "An unknown vehicle returns an empty page rather than 404: with no "
        "vehicle registry, \"no such vehicle\" and \"no events in this range\" "
        "are not distinguishable."
    ),
    responses=_QUERY_RESPONSES,
)
async def get_vehicle_telemetry(
    vehicle_id: str,
    time_range: Annotated[TimeRangeQuery, Query()],
    repository: Annotated[TelemetryRepository, Depends(get_telemetry_repository)],
) -> VehicleTelemetryPage:
    events = await repository.find_by_vehicle_and_time_range(
        vehicle_id, time_range.start, time_range.end, time_range.limit
    )
    return VehicleTelemetryPage.from_stored(
        vehicle_id, time_range.start, time_range.end, events
    )


@router.get(
    "/{vehicle_id}/anomalies",
    response_model=VehicleTelemetryPage,
    summary="Anomalous telemetry events for one vehicle",
    description=(
        "Same parameters and ordering as the telemetry history, restricted to "
        "events a **completed** inference run scored as anomalous.\n\n"
        "Events that were never scored (`PENDING`) and events whose inference "
        "did not complete (`FAILED`) are never returned here: the absence of a "
        "verdict is not a negative verdict.\n\n"
        "Inference is not implemented yet, so events ingested today are all "
        "`PENDING` and this endpoint correctly returns an empty page."
    ),
    responses=_QUERY_RESPONSES,
)
async def get_vehicle_anomalies(
    vehicle_id: str,
    time_range: Annotated[TimeRangeQuery, Query()],
    repository: Annotated[TelemetryRepository, Depends(get_telemetry_repository)],
) -> VehicleTelemetryPage:
    events = await repository.find_anomalies_by_vehicle_and_time_range(
        vehicle_id, time_range.start, time_range.end, time_range.limit
    )
    return VehicleTelemetryPage.from_stored(
        vehicle_id, time_range.start, time_range.end, events
    )
