"""Shared fixtures for the Telemetry Service suite."""

import copy
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

TELEMETRY_URL = "/api/v1/telemetry"


@pytest.fixture
def client() -> TestClient:
    """A test client over a freshly built application."""
    return TestClient(create_app())


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
