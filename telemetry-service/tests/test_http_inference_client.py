"""The HTTP inference client, against a controlled transport.

`httpx.MockTransport` runs the real client — real request building, real status
handling, real response parsing — without a socket. Only the far end is faked.
"""

import json
from typing import Callable

import httpx
import pytest

from app.application.errors import InferenceFailedError
from app.domain.inference import InferenceErrorCode, InferenceStatus
from app.domain.units import MetricName
from app.infrastructure.http_inference_client import HttpInferenceClient, create_inference_client

pytestmark = pytest.mark.anyio

BASE_URL = "http://inference.internal:8001"

CANONICAL_FEATURES = {
    MetricName.SOC: 78.5,
    MetricName.BATTERY_VOLTAGE: 396.2,
    MetricName.BATTERY_CURRENT: -14.7,
    MetricName.BATTERY_TEMPERATURE: 35.7778,
    MetricName.SPEED: 51.9818,
    MetricName.MOTOR_RPM: 4120.0,
}

VALID_BODY = {
    "is_anomaly": True,
    "anomaly_score": 0.1029,
    "model_name": "isolation-forest-telemetry",
    "model_version": "0.1.0",
}


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> HttpInferenceClient:
    transport = httpx.MockTransport(handler)
    return HttpInferenceClient(
        httpx.AsyncClient(base_url=BASE_URL, transport=transport, timeout=httpx.Timeout(2.0))
    )


def responding(status: int, body=None, *, text: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=body)

    return handler


def raising(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


async def test_it_posts_to_the_predict_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_BODY)

    await client_for(handler).predict(CANONICAL_FEATURES)

    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"{BASE_URL}/predict"


async def test_the_payload_is_exactly_the_canonical_feature_contract() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=VALID_BODY)

    await client_for(handler).predict(CANONICAL_FEATURES)

    assert seen[0] == {
        "features": {
            "soc": 78.5,
            "battery_voltage": 396.2,
            "battery_current": -14.7,
            "battery_temperature": 35.7778,
            "speed": 51.9818,
            "motor_rpm": 4120.0,
        }
    }


async def test_no_source_units_identifiers_or_timestamps_are_sent() -> None:
    """The Inference Service is stateless and needs none of them."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content.decode())
        return httpx.Response(200, json=VALID_BODY)

    await client_for(handler).predict(CANONICAL_FEATURES)

    for forbidden in (
        "unit", "source_units", "mph", "degF", "event_id", "vehicle_id",
        "site_id", "event_time", "received_at", "_id",
    ):
        assert forbidden not in seen[0], forbidden


async def test_every_feature_value_is_sent_as_a_bare_number() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=VALID_BODY)

    await client_for(handler).predict(CANONICAL_FEATURES)

    for value in seen[0]["features"].values():
        assert isinstance(value, (int, float))
        assert not isinstance(value, dict)


def test_the_configured_timeout_is_applied() -> None:
    client = create_inference_client("http://example.invalid:8001", timeout_seconds=1.5)

    assert client.timeout.connect == 1.5
    assert client.timeout.read == 1.5
    assert str(client.base_url) == "http://example.invalid:8001"


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


async def test_a_valid_response_becomes_a_completed_outcome() -> None:
    outcome = await client_for(responding(200, VALID_BODY)).predict(CANONICAL_FEATURES)

    assert outcome.status is InferenceStatus.COMPLETED
    assert outcome.is_anomaly is True
    assert outcome.anomaly_score == 0.1029
    assert outcome.model_name == "isolation-forest-telemetry"
    assert outcome.model_version == "0.1.0"
    assert outcome.error_code is None


async def test_a_negative_score_is_carried_through_unchanged() -> None:
    """The score is stored as the model reported it: no clamping, no rescaling."""
    body = {**VALID_BODY, "is_anomaly": False, "anomaly_score": -0.10342873250206741}

    outcome = await client_for(responding(200, body)).predict(CANONICAL_FEATURES)

    assert outcome.anomaly_score == -0.10342873250206741
    assert outcome.is_anomaly is False


async def test_the_upstream_verdict_is_trusted_over_the_score_sign() -> None:
    """`is_anomaly` is the authoritative verdict; we do not re-derive it."""
    body = {**VALID_BODY, "is_anomaly": False, "anomaly_score": 0.5}

    outcome = await client_for(responding(200, body)).predict(CANONICAL_FEATURES)

    assert outcome.is_anomaly is False


async def test_unknown_response_fields_are_tolerated() -> None:
    body = {**VALID_BODY, "threshold": 0.0, "explanation": "future field"}

    outcome = await client_for(responding(200, body)).predict(CANONICAL_FEATURES)

    assert outcome.status is InferenceStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Failures, each mapped to one approved code
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(httpx.ConnectTimeout("timed out"), id="connect-timeout"),
        pytest.param(httpx.ReadTimeout("timed out"), id="read-timeout"),
        pytest.param(httpx.PoolTimeout("timed out"), id="pool-timeout"),
    ],
)
async def test_a_timeout_maps_to_inference_timeout(exc: Exception) -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(raising(exc)).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.TIMEOUT


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(httpx.ConnectError("refused"), id="connection-refused"),
        pytest.param(httpx.ReadError("reset"), id="read-error"),
        pytest.param(httpx.RemoteProtocolError("bad protocol"), id="protocol-error"),
    ],
)
async def test_a_transport_failure_maps_to_inference_unreachable(exc: Exception) -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(raising(exc)).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.UNREACHABLE


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_an_upstream_server_error_maps_to_inference_unavailable(status: int) -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(status, {"error": {"code": "BOOM"}})).predict(
            CANONICAL_FEATURES
        )

    assert raised.value.error_code is InferenceErrorCode.UNAVAILABLE


async def test_model_not_loaded_maps_to_inference_unavailable() -> None:
    """The Inference Service is up but has no model: still an inference outage."""
    body = {"error": {"code": "MODEL_NOT_LOADED", "message": "No model is loaded"}}

    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(503, body)).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.UNAVAILABLE


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_a_rejected_request_maps_to_invalid_response(status: int) -> None:
    """A 4xx means the two contracts disagree, not that the service is down."""
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(status, {"error": {"code": "X"}})).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.INVALID_RESPONSE


async def test_malformed_json_maps_to_invalid_response() -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(200, text="{not json at all")).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize(
    "field", ["is_anomaly", "anomaly_score", "model_name", "model_version"]
)
async def test_a_missing_required_field_maps_to_invalid_response(field: str) -> None:
    body = {key: value for key, value in VALID_BODY.items() if key != field}

    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(200, body)).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("is_anomaly", "yes", id="verdict-as-string"),
        pytest.param("is_anomaly", 1, id="verdict-as-int"),
        pytest.param("anomaly_score", "0.5", id="score-as-string"),
        pytest.param("anomaly_score", None, id="score-null"),
        pytest.param("model_name", 42, id="name-as-int"),
        pytest.param("model_version", None, id="version-null"),
    ],
)
async def test_a_wrongly_typed_field_maps_to_invalid_response(field: str, value) -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(200, {**VALID_BODY, field: value})).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-Infinity"])
async def test_a_non_finite_score_is_rejected(score: str) -> None:
    """Not a verdict, and not even valid JSON to re-emit."""
    body = json.dumps(VALID_BODY).replace('"anomaly_score": 0.1029', f'"anomaly_score": {score}')

    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(200, text=body)).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.INVALID_RESPONSE


async def test_an_empty_body_maps_to_invalid_response() -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(200, text="")).predict(CANONICAL_FEATURES)

    assert raised.value.error_code is InferenceErrorCode.INVALID_RESPONSE


# --------------------------------------------------------------------------- #
# Nothing upstream leaks
# --------------------------------------------------------------------------- #


async def test_upstream_details_never_reach_the_error_code() -> None:
    """The code is the contract; hostnames and bodies are not part of it."""
    body = {"error": {"message": "connection to db-secret-host:5432 failed"}}

    with pytest.raises(InferenceFailedError) as raised:
        await client_for(responding(500, body)).predict(CANONICAL_FEATURES)

    assert raised.value.error_code.value == "INFERENCE_UNAVAILABLE"
    assert "db-secret-host" not in str(raised.value)
    assert "inference.internal" not in str(raised.value)


async def test_httpx_exception_names_do_not_appear_in_the_error() -> None:
    with pytest.raises(InferenceFailedError) as raised:
        await client_for(raising(httpx.ConnectError("refused"))).predict(CANONICAL_FEATURES)

    assert "httpx" not in str(raised.value)
    assert "ConnectError" not in str(raised.value)
