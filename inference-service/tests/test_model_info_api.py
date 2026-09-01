"""GET /model/info, served from the artifact actually loaded."""

from fastapi.testclient import TestClient

from tests.conftest import FIXED_TRAINED_AT, MODEL_INFO_URL, error_code


def test_model_info_returns_the_loaded_artifacts_metadata(client: TestClient, model) -> None:
    body = client.get(MODEL_INFO_URL).json()

    assert body["model_name"] == model.metadata.model_name
    assert body["model_version"] == model.metadata.model_version
    assert body["artifact_sha256"] == model.metadata.artifact_sha256


def test_model_info_matches_the_declared_contract(client: TestClient) -> None:
    body = client.get(MODEL_INFO_URL).json()

    assert set(body) == {
        "model_name",
        "model_version",
        "algorithm",
        "trained_at",
        "feature_order",
        "canonical_units",
        "artifact_sha256",
        "sklearn_version",
    }


def test_model_info_names_the_estimator(client: TestClient) -> None:
    assert client.get(MODEL_INFO_URL).json()["algorithm"] == "sklearn.ensemble.IsolationForest"


def test_model_info_publishes_the_exact_feature_order(client: TestClient) -> None:
    assert client.get(MODEL_INFO_URL).json()["feature_order"] == [
        "soc",
        "battery_voltage",
        "battery_current",
        "battery_temperature",
        "speed",
        "motor_rpm",
    ]


def test_model_info_publishes_the_canonical_units(client: TestClient) -> None:
    assert client.get(MODEL_INFO_URL).json()["canonical_units"] == {
        "soc": "percent",
        "battery_voltage": "V",
        "battery_current": "A",
        "battery_temperature": "degC",
        "speed": "km/h",
        "motor_rpm": "rpm",
    }


def test_model_info_reports_the_training_timestamp(client: TestClient) -> None:
    trained_at = client.get(MODEL_INFO_URL).json()["trained_at"]

    assert trained_at.startswith(FIXED_TRAINED_AT.strftime("%Y-%m-%dT%H:%M:%S"))


def test_model_info_reports_the_real_sklearn_version(client: TestClient) -> None:
    import sklearn

    assert client.get(MODEL_INFO_URL).json()["sklearn_version"] == sklearn.__version__


def test_model_info_is_not_hard_coded(client: TestClient, model) -> None:
    """Every value tracks the artifact, so it cannot drift from what is served."""
    body = client.get(MODEL_INFO_URL).json()

    assert body["feature_order"] == list(model.metadata.feature_order)
    assert body["algorithm"] == model.metadata.algorithm


def test_model_info_refuses_when_no_model_is_loaded(unloaded_client: TestClient) -> None:
    response = unloaded_client.get(MODEL_INFO_URL)

    assert response.status_code == 503
    assert error_code(response) == "MODEL_NOT_LOADED"


def test_model_info_describes_nothing_when_no_model_is_loaded(
    unloaded_client: TestClient,
) -> None:
    body = unloaded_client.get(MODEL_INFO_URL).json()

    assert set(body) == {"error"}
    for fabricated in ("IsolationForest", "sklearn", "feature_order", "0.1.0"):
        assert fabricated not in str(body)
