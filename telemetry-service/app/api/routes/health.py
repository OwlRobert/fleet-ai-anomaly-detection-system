"""Liveness endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import SERVICE_NAME, Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Shallow liveness of this process."""

    status: str = Field(description="`ok` while the process is serving requests.")
    service: str = Field(description="Service identity.")
    version: str = Field(description="Running service version.")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness of the Telemetry Service process",
    description=(
        "Reports only that this process is running. It deliberately does not "
        "check downstream dependencies, so a dependency outage cannot make a "
        "healthy instance look dead. Dependency readiness is a separate future "
        "endpoint."
    ),
)
def get_health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.version)
