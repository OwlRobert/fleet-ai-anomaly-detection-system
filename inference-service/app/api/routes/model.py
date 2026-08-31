"""Model metadata endpoint."""

from fastapi import APIRouter

from app.api.errors import ErrorEnvelope
from app.api.schemas import ModelInfoResponse
from app.core.errors import CapabilityNotImplementedError

router = APIRouter(tags=["model"])


@router.get(
    "/model/info",
    summary="Metadata of the loaded model artifact",
    description=(
        "Publishes the identity of the served model together with the feature "
        "order and canonical units it expects, so a caller can verify it is "
        "speaking the same dialect.\n\n"
        "No model has been trained or loaded yet, so this endpoint answers "
        "**501 Not Implemented** rather than describing a model that does not "
        "exist. The 200 schema below is the contract it will fulfil."
    ),
    responses={
        200: {"model": ModelInfoResponse, "description": "Metadata of the loaded artifact (not yet implemented)."},
        501: {"model": ErrorEnvelope, "description": "No model artifact exists in this phase."},
    },
)
def get_model_info() -> None:
    raise CapabilityNotImplementedError(
        capability="Model metadata",
        arrives_in="training and artifact loading arrive in a later phase",
    )
