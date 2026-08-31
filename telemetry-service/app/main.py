"""Telemetry Service application factory."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import health, telemetry, vehicles
from app.core.config import Settings, get_settings
from app.infrastructure.mongo import create_client, ensure_indexes, get_collection
from app.infrastructure.telemetry_repository import MongoTelemetryRepository

DESCRIPTION = """
Ingestion and query API for multinational EV fleet telemetry.

Telemetry arrives in **source units** — each metric carries its own explicit
unit, because a single vehicle may report `mph` for speed and `degC` for
temperature. `event_time` must be timezone-aware; naive timestamps are rejected
rather than assumed. Everything is normalized to canonical units and UTC before
storage, and ingestion is idempotent on `event_id`.

**Implementation status.** Validation, normalization and persistence are
implemented. Inference is not, so a stored event carries
`inference.status: "PENDING"` — stored but never scored. No anomaly verdict is
invented, and the anomalies endpoint correctly returns nothing for such events.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the MongoDB client for the life of the process.

    One client, created at startup and closed at shutdown. Indexes are ensured
    here so a fresh database is usable immediately; ``create_index`` is
    idempotent, so this is safe on every boot.

    Startup does not fail if MongoDB is unreachable: the process still serves
    `/health` and reports itself unready on `/health/ready`, and ingestion fails
    closed with `503` until the store comes back.
    """
    settings: Settings = get_settings()
    client = create_client(settings.mongodb_uri, settings.mongodb_timeout_ms)
    collection = get_collection(client[settings.mongodb_database], settings.mongodb_telemetry_collection)

    app.state.mongo_client = client
    app.state.telemetry_repository = MongoTelemetryRepository(collection)
    try:
        await ensure_indexes(collection)
    except Exception:  # noqa: BLE001 - an unreachable store must not stop the process
        app.state.indexes_ready = False
    else:
        app.state.indexes_ready = True

    try:
        yield
    finally:
        await client.close()


def create_app(*, lifespan_handler=lifespan) -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Fleet Telemetry Service",
        version=settings.version,
        summary="Telemetry ingestion and per-vehicle history for EV fleets.",
        description=DESCRIPTION,
        lifespan=lifespan_handler,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(telemetry.router)
    app.include_router(vehicles.router)
    return app


app = create_app()
