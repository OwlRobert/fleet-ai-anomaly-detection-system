"""Telemetry Service application factory."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import health, telemetry, vehicles
from app.core.config import get_settings

DESCRIPTION = """
Ingestion and query API for multinational EV fleet telemetry.

Telemetry arrives in **source units** — each metric carries its own explicit
unit, because a single vehicle may report `mph` for speed and `degC` for
temperature. `event_time` must be timezone-aware; naive timestamps are rejected
rather than assumed.

**Implementation status.** This phase establishes the contracts. Requests are
fully validated, but normalization, inference and persistence are not
implemented, so endpoints that depend on them answer `501 Not Implemented`
instead of returning fabricated data. The success schemas below are the
contracts those endpoints will fulfil.
"""


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Fleet Telemetry Service",
        version=settings.version,
        summary="Telemetry ingestion and per-vehicle history for EV fleets.",
        description=DESCRIPTION,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(telemetry.router)
    app.include_router(vehicles.router)
    return app


app = create_app()
