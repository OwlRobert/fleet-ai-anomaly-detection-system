"""Liveness and readiness."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field

from app.core.config import SERVICE_NAME, Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Shallow liveness of this process."""

    status: str = Field(description="`ok` while the process is serving requests.")
    service: str = Field(description="Service identity.")
    version: str = Field(description="Running service version.")


class DependencyStatus(BaseModel):
    """Whether one dependency is usable right now."""

    name: str = Field(description="Dependency identity.")
    ready: bool = Field(description="True when the dependency answered.")
    detail: str | None = Field(default=None, description="Why it is not ready, when it is not.")


class ReadinessResponse(BaseModel):
    """Whether this instance can actually serve traffic."""

    status: str = Field(description="`ready` or `not_ready`.")
    service: str
    version: str
    dependencies: list[DependencyStatus]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness of the Telemetry Service process",
    description=(
        "Reports only that this process is running. It deliberately does not "
        "check dependencies, so a dependency outage cannot make a healthy "
        "instance look dead, and a restart loop cannot be triggered by a "
        "database blip. Use `/health/ready` to decide whether to send traffic."
    ),
)
def get_health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.version)


async def _check_telemetry_store(request: Request) -> DependencyStatus:
    """Ping the telemetry store, reporting failure without leaking its address."""
    client = getattr(request.app.state, "mongo_client", None)
    if client is None:
        return DependencyStatus(name="telemetry_store", ready=False, detail="not configured")
    try:
        await client.admin.command("ping")
    except Exception:  # noqa: BLE001 - any driver failure means "not ready"
        return DependencyStatus(name="telemetry_store", ready=False, detail="unreachable")
    return DependencyStatus(name="telemetry_store", ready=True)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness of the Telemetry Service",
    description=(
        "Checks the dependencies ingestion cannot run without. Persistence is "
        "required — ingestion is fail-closed on it — so an unreachable "
        "telemetry store makes this instance **not ready** and answers `503`.\n\n"
        "The Inference Service is deliberately absent from this check: inference "
        "failure is fail-open for telemetry persistence, so it must never make "
        "this instance look unready. It is also not implemented yet."
    ),
    responses={
        200: {"model": ReadinessResponse, "description": "Every required dependency answered."},
        503: {"model": ReadinessResponse, "description": "A required dependency is unavailable."},
    },
)
async def get_readiness(
    request: Request,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadinessResponse:
    dependencies = [await _check_telemetry_store(request)]
    ready = all(dependency.ready for dependency in dependencies)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        service=SERVICE_NAME,
        version=settings.version,
        dependencies=dependencies,
    )
