"""POST /predict through the HTTP boundary, against a real model."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ANOMALOUS_SAMPLE, NORMAL_SAMPLE, PREDICT_URL, error_code


def test_a_normal_sample_is_scored(client: TestClient, canonical_request: dict[str, Any]) -> None:
    response = client.post(PREDICT_URL, json=canonical_request)

    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is False
    assert body["anomaly_score"] < 0


def test_an_extreme_sample_is_scored_as_anomalous(client: TestClient) -> None:
    """A synthetic anomaly demonstration, not a claim about a real vehicle fault."""
    response = client.post(PREDICT_URL, json={"features": dict(ANOMALOUS_SAMPLE)})

    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is True
    assert body["anomaly_score"] > 0


def test_the_response_carries_the_real_model_identity(
    client: TestClient, canonical_request: dict[str, Any]
) -> None:
    body = client.post(PREDICT_URL, json=canonical_request).json()
    info = client.get("/model/info").json()

    assert body["model_name"] == info["model_name"]
    assert body["model_version"] == info["model_version"]


def test_the_response_matches_the_declared_contract(
    client: TestClient, canonical_request: dict[str, Any]
) -> None:
    body = client.post(PREDICT_URL, json=canonical_request).json()

    assert set(body) == {"is_anomaly", "anomaly_score", "model_name", "model_version"}
    assert isinstance(body["is_anomaly"], bool)
    assert isinstance(body["anomaly_score"], float)


def test_the_score_is_finite(client: TestClient, canonical_request: dict[str, Any]) -> None:
    import math

    assert math.isfinite(client.post(PREDICT_URL, json=canonical_request).json()["anomaly_score"])


def test_key_order_in_the_request_cannot_change_the_result(
    client: TestClient, canonical_request: dict[str, Any]
) -> None:
    """JSON object order is not feature order."""
    reversed_keys = {"features": dict(reversed(list(NORMAL_SAMPLE.items())))}

    first = client.post(PREDICT_URL, json=canonical_request).json()
    second = client.post(PREDICT_URL, json=reversed_keys).json()

    assert first == second


def test_validation_behaviour_is_preserved(
    client: TestClient, canonical_request: dict[str, Any], request_with
) -> None:
    """A source measurement object is still rejected, model or no model."""
    body = request_with(
        canonical_request, lambda b: b["features"].__setitem__("speed", {"value": 32.3, "unit": "mph"})
    )

    response = client.post(PREDICT_URL, json=body)

    assert response.status_code == 422
    assert error_code(response) == "SCHEMA_VALIDATION_FAILED"


def test_validation_runs_before_the_model_is_consulted(unloaded_client: TestClient) -> None:
    """A malformed request is a 422 even when no model is loaded."""
    assert unloaded_client.post(PREDICT_URL, json={}).status_code == 422


# --------------------------------------------------------------------------- #
# No model loaded
# --------------------------------------------------------------------------- #


def test_predict_refuses_when_no_model_is_loaded(
    unloaded_client: TestClient, canonical_request: dict[str, Any]
) -> None:
    response = unloaded_client.post(PREDICT_URL, json=canonical_request)

    assert response.status_code == 503
    assert error_code(response) == "MODEL_NOT_LOADED"


def test_no_verdict_is_fabricated_when_the_model_is_missing(
    unloaded_client: TestClient, canonical_request: dict[str, Any]
) -> None:
    body = unloaded_client.post(PREDICT_URL, json=canonical_request).json()

    assert set(body) == {"error"}
    for fabricated in ("is_anomaly", "anomaly_score", "model_name", "model_version"):
        assert fabricated not in str(body)


@pytest.mark.parametrize("payload", [{"features": dict(NORMAL_SAMPLE)}, {}])
def test_errors_never_leak_a_path_or_a_traceback(
    unloaded_client: TestClient, payload: dict[str, Any]
) -> None:
    response = unloaded_client.post(PREDICT_URL, json=payload)

    for leak in ("Traceback", "joblib", ".py", "sklearn", "IsolationForest"):
        assert leak not in response.text
