"""Shared fixtures for the Inference Service suite."""

import copy
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

PREDICT_URL = "/predict"


@pytest.fixture
def client() -> TestClient:
    """A test client over a freshly built application."""
    return TestClient(create_app())


@pytest.fixture
def canonical_request() -> dict[str, Any]:
    """A feature vector already converted to canonical units by the caller."""
    return {
        "features": {
            "soc": 78.5,
            "battery_voltage": 396.2,
            "battery_current": -14.7,
            "battery_temperature": 35.7778,
            "speed": 51.9818,
            "motor_rpm": 4120.0,
        }
    }


@pytest.fixture
def request_with() -> Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]:
    """Return a copy of a request with one mutation applied."""

    def _apply(body: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        mutated = copy.deepcopy(body)
        mutate(mutated)
        return mutated

    return _apply


def error_code(response: Any) -> str:
    """The primary contract error code of an error response."""
    return response.json()["error"]["code"]
