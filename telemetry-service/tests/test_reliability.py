"""Reliability guarantees that are easy to break and expensive to lose.

The failure-isolation behaviour itself is covered by `test_inference_integration.py`.
What is tested here is the machinery underneath it: that the configured timeout
is really enforced against a slow server, that long-lived clients are closed,
and that an unforeseen error still answers in the contract's shape.
"""

import asyncio
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_telemetry_repository
from app.application.errors import InferenceFailedError
from app.domain.inference import InferenceErrorCode
from app.domain.units import MetricName
from app.infrastructure.http_inference_client import HttpInferenceClient, create_inference_client
from app.main import create_app

pytestmark = pytest.mark.anyio

FEATURES = {metric: 1.0 for metric in MetricName}


# --------------------------------------------------------------------------- #
# The timeout is real, not just configured
# --------------------------------------------------------------------------- #


async def test_a_server_that_never_answers_times_out() -> None:
    """A real socket that completes the handshake and then says nothing.

    A listening socket whose backlog accepts the connection but which never
    reads or replies. The existing client tests inject `httpx.ReadTimeout`,
    which proves the *mapping*; this proves the configured timeout is actually
    enforced, so a slow Inference Service cannot hold an ingest request open
    indefinitely.
    """
    silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    port = silent.getsockname()[1]

    client = create_inference_client(f"http://127.0.0.1:{port}", timeout_seconds=0.3)
    try:
        with pytest.raises(InferenceFailedError) as raised:
            await asyncio.wait_for(HttpInferenceClient(client).predict(FEATURES), timeout=10)
    finally:
        await client.aclose()
        silent.close()

    assert raised.value.error_code is InferenceErrorCode.TIMEOUT


async def test_the_timeout_is_finite_and_configurable() -> None:
    client = create_inference_client("http://example.invalid:8001", timeout_seconds=1.25)

    try:
        assert client.timeout.connect == 1.25
        assert client.timeout.read == 1.25
        assert client.timeout.write == 1.25
        assert client.timeout.pool == 1.25
    finally:
        await client.aclose()


async def test_no_retry_is_attempted() -> None:
    """One attempt per ingest. A retry here would multiply tail latency."""
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(
        base_url="http://inference.internal", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(InferenceFailedError):
            await HttpInferenceClient(client).predict(FEATURES)
    finally:
        await client.aclose()

    assert len(attempts) == 1


# --------------------------------------------------------------------------- #
# Long-lived clients are closed
# --------------------------------------------------------------------------- #


def test_the_inference_http_client_is_closed_on_shutdown() -> None:
    """One client per process, and it does not leak when the process stops."""
    app = create_app()

    with TestClient(app):
        client = app.state.inference_http_client
        assert client.is_closed is False

    assert client.is_closed is True


def test_the_clients_are_created_once_not_per_request() -> None:
    app = create_app()

    with TestClient(app) as test_client:
        first = app.state.inference_http_client
        for _ in range(3):
            test_client.get("/health")
        assert app.state.inference_http_client is first


# --------------------------------------------------------------------------- #
# An unforeseen error still answers in the contract's shape
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _no_external_services(app):
    app.state.mongo_client = None
    app.state.telemetry_repository = None
    app.state.inference_port = None
    yield


class ExplodingRepository:
    """Fails in a way no handler anticipates, carrying details that must not leak."""

    _message = "connection to db-prod-7.internal:27017 refused for user svc_telemetry"

    async def find_by_vehicle_and_time_range(self, *args, **kwargs):
        raise RuntimeError(self._message)

    async def find_anomalies_by_vehicle_and_time_range(self, *args, **kwargs):
        raise RuntimeError(self._message)


def _client_with_exploding_repository() -> TestClient:
    app = create_app(lifespan_handler=_no_external_services)
    app.dependency_overrides[get_telemetry_repository] = ExplodingRepository
    return TestClient(app, raise_server_exceptions=False)


RANGE = {"start": "2026-08-01T00:00:00Z", "end": "2026-12-01T00:00:00Z"}


def test_an_unhandled_error_uses_the_contract_error_envelope() -> None:
    response = _client_with_exploding_repository().get(
        "/api/v1/vehicles/veh-tw-0142/telemetry", params=RANGE
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


def test_an_unhandled_error_leaks_nothing_about_the_infrastructure() -> None:
    response = _client_with_exploding_repository().get(
        "/api/v1/vehicles/veh-tw-0142/telemetry", params=RANGE
    )

    for leak in (
        "db-prod-7", "svc_telemetry", "27017", "RuntimeError",
        "Traceback", "pymongo", "httpx", "/app/", ".py",
    ):
        assert leak not in response.text, leak
