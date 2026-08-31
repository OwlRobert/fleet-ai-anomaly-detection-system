"""The error envelope and the exception handlers that produce it.

Same envelope as the Telemetry Service:

    {"error": {"code": ..., "message": ..., "details": {...}}}

Only the codes this service can currently emit are defined. ``MODEL_NOT_LOADED``
arrives with model loading, which does not exist yet.
"""

from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.errors import CapabilityNotImplementedError

SCHEMA_VALIDATION_FAILED: Final = "SCHEMA_VALIDATION_FAILED"

#: Phase 1 only. Removed once a model artifact can actually be loaded and served.
NOT_IMPLEMENTED: Final = "NOT_IMPLEMENTED"


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


async def _handle_not_implemented(_: Request, exc: CapabilityNotImplementedError) -> JSONResponse:
    """Render an unimplemented capability as 501 rather than a fabricated result."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=_envelope(
            NOT_IMPLEMENTED,
            str(exc),
            {"capability": exc.capability, "arrives_in": exc.arrives_in},
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the envelope handlers to the application."""
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(CapabilityNotImplementedError, _handle_not_implemented)
