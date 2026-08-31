"""Endpoint behaviour: honest refusal, never a fabricated model or verdict."""

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import PREDICT_URL, error_code


def test_predict_is_refused_rather_than_faked(client: TestClient, canonical_request: dict[str, Any]) -> None:
    response = client.post(PREDICT_URL, json=canonical_request)

    assert response.status_code == 501
    assert error_code(response) == "NOT_IMPLEMENTED"


def test_predict_never_fabricates_a_verdict(client: TestClient, canonical_request: dict[str, Any]) -> None:
    body = client.post(PREDICT_URL, json=canonical_request).json()

    assert set(body) == {"error"}
    for fabricated in ("is_anomaly", "anomaly_score", "model_name", "model_version"):
        assert fabricated not in str(body)


def test_model_info_is_refused_rather_than_faked(client: TestClient) -> None:
    response = client.get("/model/info")

    assert response.status_code == 501
    assert error_code(response) == "NOT_IMPLEMENTED"


def test_model_info_never_describes_a_model_that_does_not_exist(client: TestClient) -> None:
    body = client.get("/model/info").json()

    assert set(body) == {"error"}
    for fabricated in ("isolation", "IsolationForest", "sklearn", "joblib", "trained_at"):
        assert fabricated not in str(body)


def test_error_responses_use_the_contract_envelope(client: TestClient) -> None:
    body = client.post(PREDICT_URL, json={}).json()

    assert set(body["error"]) == {"code", "message", "details"}
    assert {"code", "field", "message"} == set(body["error"]["details"]["errors"][0])


def test_error_responses_never_leak_a_stack_trace(
    client: TestClient, canonical_request: dict[str, Any]
) -> None:
    for response in (
        client.post(PREDICT_URL, json=canonical_request),
        client.post(PREDICT_URL, json={}),
        client.get("/model/info"),
    ):
        assert "Traceback" not in response.text
        assert "app/" not in response.text
