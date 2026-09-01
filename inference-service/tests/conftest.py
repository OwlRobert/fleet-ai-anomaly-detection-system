"""Shared fixtures for the Inference Service suite.

The artifact is trained **once per session** with the production hyperparameters
and written to a temporary file, so the tests exercise the same model and the
same loading path the service uses — no stub estimator, no in-memory fake.
"""

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import joblib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.inference_service import InferenceService
from app.infrastructure.artifact import LoadedArtifact, load_artifact
from app.infrastructure.isolation_forest_model import IsolationForestAnomalyModel
from app.main import create_app
from ml.train import train

PREDICT_URL = "/predict"
MODEL_INFO_URL = "/model/info"

FIXED_TRAINED_AT = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
"""Pinned so an identical training run produces an identical artifact."""

NORMAL_SAMPLE: dict[str, float] = {
    "soc": 62.0,
    "battery_voltage": 378.0,
    "battery_current": -190.0,
    "battery_temperature": 29.0,
    "speed": 63.0,
    "motor_rpm": 5170.0,
}
"""Steady cruising, consistent with how the training data is generated."""

ANOMALOUS_SAMPLE: dict[str, float] = {
    "soc": 80.0,
    "battery_voltage": 250.0,
    "battery_current": -900.0,
    "battery_temperature": 85.0,
    "speed": 10.0,
    "motor_rpm": 800.0,
}
"""Far outside normal operation on several axes at once.

A synthetic demonstration of an anomaly, not a claim about any real vehicle
fault.
"""


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real artifact, trained once with the production configuration."""
    path = tmp_path_factory.mktemp("artifacts") / "isolation_forest_test.joblib"
    joblib.dump(train(trained_at=FIXED_TRAINED_AT), path)
    return path


@pytest.fixture(scope="session")
def loaded_artifact(artifact_path: Path) -> LoadedArtifact:
    """The artifact, read back through the production loader."""
    return load_artifact(artifact_path)


@pytest.fixture(scope="session")
def model(loaded_artifact: LoadedArtifact) -> IsolationForestAnomalyModel:
    """The model itself is stateless, so one instance serves the whole session."""
    return IsolationForestAnomalyModel(loaded_artifact)


@pytest.fixture(scope="session")
def service(model: IsolationForestAnomalyModel) -> InferenceService:
    return InferenceService(model=model)


def _app_with(service: InferenceService) -> FastAPI:
    """An application whose startup installs the given service."""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        app.state.inference_service = service
        yield

    return create_app(lifespan_handler=lifespan)


@pytest.fixture
def app(service: InferenceService) -> FastAPI:
    return _app_with(service)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """A client over an application with a real model loaded."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def unloaded_client() -> Iterator[TestClient]:
    """A client over an application whose artifact failed to load."""
    with TestClient(_app_with(InferenceService(model=None))) as test_client:
        yield test_client


@pytest.fixture
def canonical_request() -> dict[str, Any]:
    """A feature vector already converted to canonical units by the caller."""
    return {"features": dict(NORMAL_SAMPLE)}


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
