"""The error envelope and the exception handlers that produce it.

Same envelope as the Telemetry Service:

    {"error": {"code": ..., "message": ..., "details": {...}}}

Only the codes this service can currently emit are defined. ``MODEL_NOT_LOADED``
arrives with model loading, which does not exist yet.
"""

import logging
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.errors import ModelNotLoadedError

logger = logging.getLogger(__name__)

SCHEMA_VALIDATION_FAILED: Final = "SCHEMA_VALIDATION_FAILED"
MODEL_NOT_LOADED: Final = "MODEL_NOT_LOADED"

#: Last resort, so an unforeseen failure still answers in the contract's shape.
INTERNAL_ERROR: Final = "INTERNAL_ERROR"


class Error(BaseModel):
    """Body of the error envelope."""

    code: str = Field(description="Primary contract error code for this response.")
    message: str = Field(description="Human-readable summary. Not part of the contract.")
    details: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured context. Validation failures carry an `errors` array of "
            "`{code, field, message}` objects, one per offending field."
        ),
    )


class ErrorEnvelope(BaseModel):
    """The single error shape used by every non-2xx response."""

    error: Error

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "SCHEMA_VALIDATION_FAILED",
                    "message": "Request failed the canonical feature contract.",
                    "details": {
                        "errors": [
                            {
                                "code": "SCHEMA_VALIDATION_FAILED",
                                "field": "body.features.speed",
                                "message": "Input should be a valid number",
                            }
                        ]
                    },
                }
            }
        }
    }


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Render Pydantic validation failures in the contract envelope."""
    errors = [
        {
            "code": SCHEMA_VALIDATION_FAILED,
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_envelope(
            SCHEMA_VALIDATION_FAILED,
            "Request failed the canonical feature contract.",
            {"errors": errors},
        ),
    )


async def _handle_model_not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """Say the model is unavailable rather than inventing a verdict.

    The message is fixed prose. The exception text and the artifact path stay in
    the logs: a client has no business learning where the file lives.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_envelope(
            MODEL_NOT_LOADED,
            "No model is loaded; this instance cannot serve predictions.",
            {"retryable": True},
        ),
    )


async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """Anything unforeseen becomes the same envelope every other error uses.

    The exception and its traceback go to the log; the client gets a code and
    nothing else. Without this, an unhandled error would answer with a bare
    ``Internal Server Error`` string, breaking the one error shape every other
    response keeps.
    """
    logger.exception("unhandled error while serving a request")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope(INTERNAL_ERROR, "An unexpected internal error occurred."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the envelope handlers to the application."""
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(ModelNotLoadedError, _handle_model_not_loaded)
    app.add_exception_handler(Exception, _handle_unexpected_error)
