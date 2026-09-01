"""Shared fixtures for the Telemetry Service suite."""

import copy
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_inference_port,
    get_telemetry_normalizer,
    get_telemetry_repository,
)
from app.core.config import get_settings
from app.domain.normalizer import TelemetryNormalizer
from app.main import create_app
from tests.fakes import InMemoryTelemetryRepository, StubInferencePort

TELEMETRY_URL = "/api/v1/telemetry"

FIXED_NOW = datetime(2026, 9, 1, 16, 0, 0, tzinfo=timezone.utc)
"""The server clock every API test runs against.

Normalization stamps ``received_at`` and bounds clock skew against it, so the
endpoint's behaviour would otherwise drift with the wall clock. Pinning it keeps
every assertion exact and keeps the suite passing a year from now.
"""


@pytest.fixture
def anyio_backend() -> str:
    """Run `@pytest.mark.anyio` tests on asyncio. No extra plugin needed."""
    return "asyncio"


@asynccontextmanager
async def _no_external_services(app) -> AsyncIterator[None]:
    """Replace the real lifespan: no MongoDB connection, no HTTP client."""
    app.state.mongo_client = None
    app.state.telemetry_repository = None
    app.state.inference_port = None
    yield


@pytest.fixture
def repository() -> InMemoryTelemetryRepository:
    """The store the API tests run against."""
    return InMemoryTelemetryRepository()


@pytest.fixture
def inference() -> StubInferencePort:
    """The inference boundary the API tests run against."""
    return StubInferencePort()


@pytest.fixture
def app(repository: InMemoryTelemetryRepository, inference: StubInferencePort) -> FastAPI:
    """An application on the fixed clock, the in-memory store and a stub model."""
    application = create_app(lifespan_handler=_no_external_services)
    settings = get_settings()
    application.dependency_overrides[get_telemetry_normalizer] = lambda: TelemetryNormalizer(
        clock=lambda: FIXED_NOW,
        max_future_skew=timedelta(seconds=settings.max_clock_skew_future_seconds),
        max_event_age=timedelta(days=settings.max_event_age_days),
    )
    application.dependency_overrides[get_telemetry_repository] = lambda: repository
    application.dependency_overrides[get_inference_port] = lambda: inference
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """A test client whose normalizer runs on the fixed clock."""
    return TestClient(app)


@pytest.fixture
def live_client(
    repository: InMemoryTelemetryRepository, inference: StubInferencePort
) -> TestClient:
    """Real clock and real settings, but no database and no HTTP calls."""
    application = create_app(lifespan_handler=_no_external_services)
    application.dependency_overrides[get_telemetry_repository] = lambda: repository
    application.dependency_overrides[get_inference_port] = lambda: inference
    return TestClient(application)


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """A telemetry event that satisfies the whole contract.

    Deliberately mixes unit systems — mph for speed, degF for temperature —
    because units are per metric, not per request.
    """
    return {
        "schema_version": "1.0",
        "event_id": "3f0a9c2e-6f4b-4a6f-9d6e-2b1c8f4a77e1",
        "vehicle_id": "veh-tw-0142",
        "site_id": "site-taipei-01",
        "event_time": "2026-08-31T09:14:22.481+08:00",
        "metrics": {
            "soc": {"value": 78.5, "unit": "percent"},
            "battery_voltage": {"value": 396.2, "unit": "V"},
            "battery_current": {"value": -14.7, "unit": "A"},
            "battery_temperature": {"value": 96.4, "unit": "degF"},
            "speed": {"value": 32.3, "unit": "mph"},
            "motor_rpm": {"value": 4120, "unit": "rpm"},
        },
    }


@pytest.fixture
def payload_with() -> Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]:
    """Return a copy of a payload with one mutation applied."""

    def _apply(payload: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        mutated = copy.deepcopy(payload)
        mutate(mutated)
        return mutated

    return _apply


def error_code(response: Any) -> str:
    """The primary contract error code of an error response."""
    return response.json()["error"]["code"]
