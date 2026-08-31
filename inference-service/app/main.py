"""Inference Service application factory."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import health, model, prediction
from app.core.config import get_settings

DESCRIPTION = """
Anomaly scoring for EV fleet telemetry.

This service is **stateless and unit-agnostic**. It accepts canonical feature
values only — `soc` percent, `battery_voltage` V, `battery_current` A,
`battery_temperature` degC, `speed` km/h, `motor_rpm` rpm — and performs no unit
conversion. It knows nothing about vehicles, sites or clients.

**Implementation status.** This phase establishes the contracts. Requests are
fully validated, but no model has been trained or loaded, so `/predict` and
`/model/info` answer `501 Not Implemented` instead of inventing a result. The
success schemas below are the contracts they will fulfil.
"""


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Fleet Inference Service",
        version=settings.version,
        summary="Anomaly scoring over canonical EV telemetry features.",
        description=DESCRIPTION,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(prediction.router)
    return app


app = create_app()
