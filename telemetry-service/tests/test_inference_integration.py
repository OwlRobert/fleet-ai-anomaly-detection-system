"""Ingestion end to end through the API, with inference in the path.

Covers the success flow, the fail-open policy for every approved failure code,
and the interaction with idempotency and persistence failure.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.domain.inference import InferenceErrorCode, InferenceStatus
from app.domain.units import MetricName
from tests.conftest import TELEMETRY_URL, error_code
from tests.fakes import InMemoryTelemetryRepository, StubInferencePort, UnavailableTelemetryRepository

FAILURE_CODES = [
    pytest.param(InferenceErrorCode.TIMEOUT, id="timeout"),
    pytest.param(InferenceErrorCode.UNREACHABLE, id="connection-failure"),
    pytest.param(InferenceErrorCode.UNAVAILABLE, id="upstream-5xx-or-model-unavailable"),
    pytest.param(InferenceErrorCode.INVALID_RESPONSE, id="malformed-or-invalid-contract"),
]


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #


def test_a_scored_event_is_created(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 201
    assert response.json()["inference"]["status"] == "COMPLETED"


def test_the_models_values_are_persisted_exactly(
    app, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository
) -> None:
    from app.api.dependencies import get_inference_port

    model = StubInferencePort(
        is_anomaly=True, anomaly_score=0.1029, model_name="isolation-forest-telemetry",
        model_version="0.1.0",
    )
    app.dependency_overrides[get_inference_port] = lambda: model

    TestClient(app).post(TELEMETRY_URL, json=valid_payload)

    stored = repository.events[0].inference
    assert stored.status is InferenceStatus.COMPLETED
    assert stored.is_anomaly is True
    assert stored.anomaly_score == 0.1029
    assert stored.model_name == "isolation-forest-telemetry"
    assert stored.model_version == "0.1.0"
    assert stored.error_code is None


@pytest.mark.parametrize("is_anomaly", [True, False])
def test_both_verdicts_round_trip(
    app, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository, is_anomaly: bool
) -> None:
    from app.api.dependencies import get_inference_port

    app.dependency_overrides[get_inference_port] = lambda: StubInferencePort(
        is_anomaly=is_anomaly, anomaly_score=0.5 if is_anomaly else -0.5
    )

    body = TestClient(app).post(TELEMETRY_URL, json=valid_payload).json()

    assert body["inference"]["is_anomaly"] is is_anomaly
    assert repository.events[0].inference.is_anomaly is is_anomaly


def test_only_canonical_values_cross_the_inference_boundary(
    client: TestClient, valid_payload: dict[str, Any], inference: StubInferencePort
) -> None:
    """mph and degF go in; km/h and degC are what the model is asked about."""
    client.post(TELEMETRY_URL, json=valid_payload)

    sent = inference.calls[0]
    assert sent[MetricName.SPEED] == pytest.approx(51.9818112)
    assert sent[MetricName.BATTERY_TEMPERATURE] == pytest.approx(35.7777777777)
    assert set(sent) == set(MetricName)
    assert all(isinstance(value, float) for value in sent.values())


def test_the_model_is_called_once_per_new_event(
    client: TestClient, valid_payload: dict[str, Any], inference: StubInferencePort
) -> None:
    client.post(TELEMETRY_URL, json=valid_payload)

    assert inference.call_count == 1


def test_a_synchronous_ingest_never_finishes_pending(
    client: TestClient, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository
) -> None:
    client.post(TELEMETRY_URL, json=valid_payload)

    assert repository.events[0].inference.status is not InferenceStatus.PENDING


# --------------------------------------------------------------------------- #
# Fail-open: inference failure never costs the measurement
# --------------------------------------------------------------------------- #


def failing_app(app, code: InferenceErrorCode) -> TestClient:
    from app.api.dependencies import get_inference_port

    app.dependency_overrides[get_inference_port] = lambda: StubInferencePort(fail_with=code)
    return TestClient(app)


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_ingestion_still_succeeds_when_inference_fails(
    app, valid_payload: dict[str, Any], code: InferenceErrorCode
) -> None:
    response = failing_app(app, code).post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 201
    assert response.json()["duplicate"] is False


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_the_telemetry_is_persisted_when_inference_fails(
    app, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository,
    code: InferenceErrorCode,
) -> None:
    failing_app(app, code).post(TELEMETRY_URL, json=valid_payload)

    assert len(repository.events) == 1
    assert repository.events[0].event.metrics[MetricName.SPEED] == pytest.approx(51.9818112)


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_no_verdict_is_invented_when_inference_fails(
    app, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository,
    code: InferenceErrorCode,
) -> None:
    """Not `is_anomaly: false`, not a score of 0, not a model name."""
    failing_app(app, code).post(TELEMETRY_URL, json=valid_payload)

    stored = repository.events[0].inference
    assert stored.status is InferenceStatus.FAILED
    assert stored.is_anomaly is None
    assert stored.anomaly_score is None
    assert stored.model_name is None
    assert stored.model_version is None
    assert stored.is_confirmed_anomaly is False


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_the_failure_reason_is_stored_and_returned(
    app, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository,
    code: InferenceErrorCode,
) -> None:
    body = failing_app(app, code).post(TELEMETRY_URL, json=valid_payload).json()

    assert body["inference"]["error_code"] == code.value
    assert repository.events[0].inference.error_code == code.value


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_an_inference_failure_is_never_reported_as_503(
    app, valid_payload: dict[str, Any], code: InferenceErrorCode
) -> None:
    """503 is reserved for persistence. Inference failure does not reject the event."""
    assert failing_app(app, code).post(TELEMETRY_URL, json=valid_payload).status_code != 503


def test_an_unscored_failure_never_appears_in_the_anomaly_query(
    app, valid_payload: dict[str, Any]
) -> None:
    client = failing_app(app, InferenceErrorCode.TIMEOUT)
    client.post(TELEMETRY_URL, json=valid_payload)

    window = {"start": "2026-08-01T00:00:00Z", "end": "2026-12-01T00:00:00Z"}
    history = client.get("/api/v1/vehicles/veh-tw-0142/telemetry", params=window).json()
    anomalies = client.get("/api/v1/vehicles/veh-tw-0142/anomalies", params=window).json()

    assert history["count"] == 1
    assert history["items"][0]["inference"]["status"] == "FAILED"
    assert anomalies["count"] == 0


def test_a_failure_response_leaks_no_upstream_detail(
    app, valid_payload: dict[str, Any]
) -> None:
    response = failing_app(app, InferenceErrorCode.UNAVAILABLE).post(
        TELEMETRY_URL, json=valid_payload
    )

    for leak in ("httpx", "Traceback", "8001", "localhost", "http://", "ConnectError"):
        assert leak not in response.text


# --------------------------------------------------------------------------- #
# Persistence stays fail-closed, whatever inference did
# --------------------------------------------------------------------------- #


def test_persistence_failure_after_successful_inference_is_503(
    app, valid_payload: dict[str, Any]
) -> None:
    from app.api.dependencies import get_telemetry_repository

    app.dependency_overrides[get_telemetry_repository] = UnavailableTelemetryRepository

    response = TestClient(app).post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 503
    assert error_code(response) == "PERSISTENCE_UNAVAILABLE"


def test_persistence_failure_after_failed_inference_is_503(
    app, valid_payload: dict[str, Any]
) -> None:
    """Two failures, and the fail-closed one decides the answer."""
    from app.api.dependencies import get_inference_port, get_telemetry_repository

    app.dependency_overrides[get_inference_port] = lambda: StubInferencePort(
        fail_with=InferenceErrorCode.TIMEOUT
    )
    app.dependency_overrides[get_telemetry_repository] = UnavailableTelemetryRepository

    response = TestClient(app).post(TELEMETRY_URL, json=valid_payload)

    assert response.status_code == 503
    assert error_code(response) == "PERSISTENCE_UNAVAILABLE"
    assert set(response.json()) == {"error"}


# --------------------------------------------------------------------------- #
# Duplicates never re-score
# --------------------------------------------------------------------------- #


def test_a_retry_of_a_scored_event_does_not_call_inference_again(
    client: TestClient, valid_payload: dict[str, Any], inference: StubInferencePort,
    repository: InMemoryTelemetryRepository,
) -> None:
    first = client.post(TELEMETRY_URL, json=valid_payload).json()

    retry = client.post(TELEMETRY_URL, json=valid_payload)

    assert retry.status_code == 200
    assert retry.json()["duplicate"] is True
    assert inference.call_count == 1
    assert len(repository.events) == 1
    assert retry.json()["inference"] == first["inference"]


def test_a_retry_of_a_failed_event_does_not_re_score_it(
    app, valid_payload: dict[str, Any], repository: InMemoryTelemetryRepository
) -> None:
    """A stored FAILED verdict is returned as-is, not retried inside the request."""
    from app.api.dependencies import get_inference_port

    model = StubInferencePort(fail_with=InferenceErrorCode.TIMEOUT)
    app.dependency_overrides[get_inference_port] = lambda: model
    client = TestClient(app)
    client.post(TELEMETRY_URL, json=valid_payload)

    model.fail_with = None  # the model recovers
    retry = client.post(TELEMETRY_URL, json=valid_payload)

    assert retry.status_code == 200
    assert retry.json()["duplicate"] is True
    assert retry.json()["inference"]["status"] == "FAILED"
    assert retry.json()["inference"]["error_code"] == "INFERENCE_TIMEOUT"
    assert model.call_count == 1
    assert len(repository.events) == 1


def test_a_conflicting_reuse_does_not_call_inference(
    client: TestClient, valid_payload: dict[str, Any], payload_with,
    inference: StubInferencePort, repository: InMemoryTelemetryRepository,
) -> None:
    client.post(TELEMETRY_URL, json=valid_payload)
    conflicting = payload_with(valid_payload, lambda p: p.__setitem__("vehicle_id", "veh-cz-0007"))

    response = client.post(TELEMETRY_URL, json=conflicting)

    assert response.status_code == 200
    assert response.json()["vehicle_id"] == "veh-tw-0142"
    assert inference.call_count == 1
    assert len(repository.events) == 1
