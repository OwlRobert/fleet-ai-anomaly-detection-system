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
    summary="Liveness of the Inference Service process",
    description=(
        "Reports only that this process is running. It reports nothing about a "
        "model, because model loading does not exist yet — a `model_loaded` "
        "field joins this response when it does, rather than being reported "
        "now as a value that means nothing."
    ),
)
def get_health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=settings.version)
