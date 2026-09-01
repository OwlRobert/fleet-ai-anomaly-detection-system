"""Telemetry Service application factory."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import health, telemetry, vehicles
from app.core.config import Settings, get_settings
from app.infrastructure.http_inference_client import (
    HttpInferenceClient,
    create_inference_client,
)
from app.infrastructure.mongo import create_client, ensure_indexes, get_collection
from app.infrastructure.telemetry_repository import MongoTelemetryRepository

DESCRIPTION = """
Ingestion and query API for multinational EV fleet telemetry.

Telemetry arrives in **source units** — each metric carries its own explicit
unit, because a single vehicle may report `mph` for speed and `degC` for
temperature. `event_time` must be timezone-aware; naive timestamps are rejected
rather than assumed. Everything is normalized to canonical units and UTC before
storage, and ingestion is idempotent on `event_id`.

Each event is scored synchronously by the Inference Service before it is
stored. The two dependencies have opposite failure policies: if inference
fails the telemetry is stored anyway with `inference.status: "FAILED"` and no
invented verdict (**fail-open**); if persistence fails the request returns
`503` and nothing is acknowledged (**fail-closed**).
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the MongoDB client for the life of the process.

    One client, created at startup and closed at shutdown. Indexes are ensured
    here so a fresh database is usable immediately; ``create_index`` is
    idempotent, so this is safe on every boot.

    The HTTP client for the Inference Service is owned the same way: one
    connection pool for the process, not one per request. Building it opens no
    connection, so an Inference Service that is down does not affect startup —
    that failure surfaces per request, fail-open.

    Startup does not fail if MongoDB is unreachable either: the process still
    serves `/health` and reports itself unready on `/health/ready`, and
    ingestion fails closed with `503` until the store comes back.
    """
    settings: Settings = get_settings()
    client = create_client(settings.mongodb_uri, settings.mongodb_timeout_ms)
    collection = get_collection(client[settings.mongodb_database], settings.mongodb_telemetry_collection)

    inference_client = create_inference_client(
        settings.inference_service_url, settings.inference_timeout_seconds
    )

    app.state.mongo_client = client
    app.state.telemetry_repository = MongoTelemetryRepository(collection)
    app.state.inference_http_client = inference_client
    app.state.inference_port = HttpInferenceClient(inference_client)
    try:
        await ensure_indexes(collection)
    except Exception:  # noqa: BLE001 - an unreachable store must not stop the process
        app.state.indexes_ready = False
    else:
        app.state.indexes_ready = True

    try:
        yield
    finally:
        await inference_client.aclose()
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
