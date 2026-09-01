"""Model metadata endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_inference_service
from app.api.errors import ErrorEnvelope
from app.api.schemas import ModelInfoResponse
from app.application.inference_service import InferenceService

router = APIRouter(tags=["model"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Metadata of the loaded model artifact",
    description=(
        "Publishes the identity of the model actually loaded by this process, "
        "together with the feature order and canonical units it expects, so a "
        "caller can verify it is speaking the same dialect.\n\n"
        "Every field is read from the loaded artifact, so it cannot drift from "
        "the model being served. `artifact_sha256` identifies the exact file.\n\n"
        "With no model loaded this answers `503 MODEL_NOT_LOADED` rather than "
        "describing a model that is not there."
    ),
    responses={
        200: {"model": ModelInfoResponse, "description": "Metadata of the loaded artifact."},
        503: {"model": ErrorEnvelope, "description": "No model is loaded."},
    },
)
def get_model_info(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> ModelInfoResponse:
    return ModelInfoResponse.from_metadata(service.describe_model())
