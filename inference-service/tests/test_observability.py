"""Logging context and request correlation for the Inference Service."""

import json
import logging

from fastapi.testclient import TestClient

from app.core.logging import JsonFormatter, configure_logging
from app.core.request_id import REQUEST_ID_HEADER


def _record_with(extra: dict) -> logging.LogRecord:
    made = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="model artifact loaded", args=None, exc_info=None,
    )
    for key, value in extra.items():
        setattr(made, key, value)
    return made


def test_json_logs_carry_the_attached_model_identifiers() -> None:
    payload = json.loads(
        JsonFormatter().format(
            _record_with({"model_name": "isolation-forest-telemetry", "model_version": "0.1.0"})
        )
    )

    assert payload["model_name"] == "isolation-forest-telemetry"
    assert payload["model_version"] == "0.1.0"
    assert payload["level"] == "INFO"


def test_configuring_logging_twice_does_not_duplicate_handlers() -> None:
    configure_logging("INFO", "json")
    configure_logging("INFO", "json")

    assert len(logging.getLogger().handlers) == 1


def test_a_request_id_is_generated_when_none_is_supplied(client: TestClient) -> None:
    assert len(client.get("/health").headers[REQUEST_ID_HEADER]) >= 8


def test_a_supplied_request_id_is_echoed_back(client: TestClient) -> None:
    """The Telemetry Service forwards its id here, so one id spans both services."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"


def test_an_oversized_request_id_is_bounded(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "x" * 5000})

    assert len(response.headers[REQUEST_ID_HEADER]) <= 128


def test_error_responses_also_carry_a_request_id(unloaded_client: TestClient) -> None:
    response = unloaded_client.get("/model/info")

    assert response.status_code == 503
    assert response.headers[REQUEST_ID_HEADER]


def test_an_unhandled_error_uses_the_contract_error_envelope() -> None:
    """Even an unforeseen failure answers in the one error shape."""
    from contextlib import asynccontextmanager

    from app.main import create_app

    class Exploding:
        def predict(self, features):
            raise RuntimeError("/app/ml/artifacts/model.joblib is corrupt at offset 4096")

        @property
        def metadata(self):
            raise RuntimeError("/app/ml/artifacts/model.joblib is corrupt at offset 4096")

    @asynccontextmanager
    async def lifespan(app):
        from app.application.inference_service import InferenceService

        app.state.inference_service = InferenceService(model=Exploding())
        yield

    with TestClient(create_app(lifespan_handler=lifespan), raise_server_exceptions=False) as client:
        response = client.get("/model/info")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    for leak in ("/app/", "joblib", "RuntimeError", "Traceback", "offset 4096"):
        assert leak not in response.text, leak
