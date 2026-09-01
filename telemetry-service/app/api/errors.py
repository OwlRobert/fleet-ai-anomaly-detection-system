"""The error envelope and the exception handlers that produce it.

Every error response uses the envelope defined in the architecture:

    {"error": {"code": ..., "message": ..., "details": {...}}}

``code`` is the machine-readable contract; ``message`` is English-only prose and
carries no contract weight. Python exceptions never reach the client.
"""

import logging
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.application.errors import CapabilityNotImplementedError, PersistenceUnavailableError

logger = logging.getLogger(__name__)
from app.domain.errors import (
    ClockSkewError,
    ClockSkewFutureError,
    EventTooOldError,
    NormalizationError,
)

# Contract error codes this service can emit.
SCHEMA_VALIDATION_FAILED: Final = "SCHEMA_VALIDATION_FAILED"
UNSUPPORTED_SCHEMA_VERSION: Final = "UNSUPPORTED_SCHEMA_VERSION"
NAIVE_TIMESTAMP: Final = "NAIVE_TIMESTAMP"
UNSUPPORTED_UNIT: Final = "UNSUPPORTED_UNIT"
UNKNOWN_METRIC: Final = "UNKNOWN_METRIC"
MISSING_METRIC: Final = "MISSING_METRIC"
INVALID_TIME_RANGE: Final = "INVALID_TIME_RANGE"
CLOCK_SKEW_FUTURE: Final = "CLOCK_SKEW_FUTURE"
EVENT_TOO_OLD: Final = "EVENT_TOO_OLD"
PERSISTENCE_UNAVAILABLE: Final = "PERSISTENCE_UNAVAILABLE"

#: Last resort, so an unforeseen failure still answers in the contract's shape.
INTERNAL_ERROR: Final = "INTERNAL_ERROR"

#: Which contract code each normalization rejection maps to. The domain raises
#: a typed error and stays free of API vocabulary; the mapping lives here.
_NORMALIZATION_CODES: Final[dict[type[NormalizationError], str]] = {
    ClockSkewFutureError: CLOCK_SKEW_FUTURE,
    EventTooOldError: EVENT_TOO_OLD,
}

#: Not part of the approved contract. Removed as each capability lands.
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
                    "code": "NAIVE_TIMESTAMP",
                    "message": "event_time must include a UTC offset or Z designator",
                    "details": {"errors": [{"code": "NAIVE_TIMESTAMP", "field": "body.event_time", "message": "Input should have timezone info"}]},
                }
            }
        }
    }


def _code_for(error: dict[str, Any]) -> str:
    """Map one Pydantic validation error onto a contract error code."""
    location = error["loc"]
    error_type = error["type"]
    field = location[-1] if location else None

    if error_type == "timezone_aware":
        return NAIVE_TIMESTAMP
    if error_type == "invalid_time_range":
        return INVALID_TIME_RANGE
    if error_type == "literal_error":
        if field == "unit":
            return UNSUPPORTED_UNIT
        if field == "schema_version":
            return UNSUPPORTED_SCHEMA_VERSION
    if "metrics" in location:
        if error_type == "missing":
            return MISSING_METRIC
        if error_type == "extra_forbidden":
            return UNKNOWN_METRIC
    return SCHEMA_VALIDATION_FAILED


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Render Pydantic validation failures in the contract envelope."""
    errors = [
        {
            "code": _code_for(error),
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    primary = errors[0]["code"] if errors else SCHEMA_VALIDATION_FAILED
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_envelope(primary, "Request failed telemetry contract validation.", {"errors": errors}),
    )


async def _handle_normalization_error(_: Request, exc: NormalizationError) -> JSONResponse:
    """Render a normalization rejection with its contract code.

    The offending timestamps are echoed back so a caller can see *why* it was
    rejected. Neither is corrected or clamped.
    """
    code = _NORMALIZATION_CODES.get(type(exc), SCHEMA_VALIDATION_FAILED)
    details: dict[str, Any] = {"field": "event_time"}
    if isinstance(exc, ClockSkewError):
        details |= {
            "event_time": exc.event_time.isoformat(),
            "received_at": exc.received_at.isoformat(),
            "limit_seconds": exc.limit.total_seconds(),
        }
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_envelope(code, str(exc), details),
    )


async def _handle_persistence_unavailable(
    _: Request, exc: PersistenceUnavailableError
) -> JSONResponse:
    """Ingestion is fail-closed on persistence: say so, and say nothing more.

    The client is told the event was **not** accepted, so its retry — safe by
    ``event_id`` idempotency — is the recovery path. Driver exception names,
    hostnames and connection strings never appear in the response.
    """
    logger.warning(
        "telemetry store unavailable; request not acknowledged",
        extra={"error_code": PERSISTENCE_UNAVAILABLE},
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_envelope(
            PERSISTENCE_UNAVAILABLE,
            "The telemetry store is unavailable; the event was not accepted. Retry is safe.",
            {"retryable": True},
        ),
    )


async def _handle_not_implemented(_: Request, exc: CapabilityNotImplementedError) -> JSONResponse:
    """Render an unimplemented capability as 501 rather than a fabricated success."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=_envelope(
            NOT_IMPLEMENTED,
            str(exc),
            {"capability": exc.capability, "arrives_in": exc.arrives_in},
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
    app.add_exception_handler(NormalizationError, _handle_normalization_error)
    app.add_exception_handler(PersistenceUnavailableError, _handle_persistence_unavailable)
    app.add_exception_handler(CapabilityNotImplementedError, _handle_not_implemented)
    app.add_exception_handler(Exception, _handle_unexpected_error)
