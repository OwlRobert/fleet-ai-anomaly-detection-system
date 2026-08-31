"""Prediction endpoint."""

from fastapi import APIRouter

from app.api.errors import ErrorEnvelope
from app.api.schemas import PredictionRequest, PredictionResponse
from app.core.errors import CapabilityNotImplementedError

router = APIRouter(tags=["inference"])


@router.post(
    "/predict",
    summary="Score one canonical feature vector",
    description=(
        "Accepts **canonical values only**: bare numbers already converted by "
        "the caller. Source measurement objects such as "
        "`{\"value\": 32.3, \"unit\": \"mph\"}`, and any `unit` or "
        "`source_units` key, are rejected — the model must never depend on a "
        "client's display preference.\n\n"
        "No model has been trained or loaded yet, so a valid request is "
        "answered with **501 Not Implemented**. No score is invented. The 200 "
        "schema below is the contract this endpoint will fulfil."
    ),
    responses={
        200: {"model": PredictionResponse, "description": "The model's verdict (not yet implemented)."},
        422: {"model": ErrorEnvelope, "description": "Request failed the canonical feature contract."},
        501: {"model": ErrorEnvelope, "description": "No model artifact exists in this phase."},
    },
)
def predict(request: PredictionRequest) -> None:
    raise CapabilityNotImplementedError(
        capability="Model inference",
        arrives_in="training and artifact loading arrive in a later phase",
    )
