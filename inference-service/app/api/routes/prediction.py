"""Prediction endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_inference_service
from app.api.errors import ErrorEnvelope
from app.api.schemas import PredictionRequest, PredictionResponse
from app.application.inference_service import InferenceService

router = APIRouter(tags=["inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score one canonical feature vector",
    description=(
        "Accepts **canonical values only**: bare numbers already converted by "
        "the caller. Source measurement objects such as "
        "`{\"value\": 32.3, \"unit\": \"mph\"}`, and any `unit` or "
        "`source_units` key, are rejected — the model must never depend on a "
        "client's display preference.\n\n"
        "`is_anomaly` is the model's own verdict. `anomaly_score` is "
        "anomaly-oriented: **higher means more anomalous**, with the decision "
        "boundary at zero. It is a ranking score, not a probability.\n\n"
        "With no model loaded this answers `503 MODEL_NOT_LOADED`. No score is "
        "ever invented."
    ),
    responses={
        200: {"model": PredictionResponse, "description": "The model's verdict."},
        422: {"model": ErrorEnvelope, "description": "Request failed the canonical feature contract."},
        503: {"model": ErrorEnvelope, "description": "No model is loaded."},
    },
)
def predict(
    request: PredictionRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> PredictionResponse:
    prediction = service.predict(request.features.model_dump())
    return PredictionResponse.from_prediction(prediction)
