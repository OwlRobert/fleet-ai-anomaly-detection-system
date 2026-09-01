"""Liveness, and whether this instance can actually score."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_inference_service
from app.application.inference_service import InferenceService
from app.core.config import SERVICE_NAME, Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness of this process, and whether a model is loaded."""

    status: str = Field(description="`ok` while the process is serving requests.")
    service: str = Field(description="Service identity.")
    version: str = Field(description="Running service version.")
    model_loaded: bool = Field(
        description="Whether a model artifact was loaded successfully at startup."
    )

    model_config = {"protected_namespaces": ()}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness of the Inference Service process",
    description=(
        "Always answers `200` while the process is alive, and reports whether a "
        "model is loaded.\n\n"
        "A failed artifact load does **not** make this endpoint fail: the "
        "process is running, it simply cannot serve predictions, and "
        "`model_loaded: false` says exactly that. `/predict` and `/model/info` "
        "answer `503 MODEL_NOT_LOADED` in that state."
    ),
)
def get_health(
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=settings.version,
        model_loaded=service.is_model_loaded,
    )
