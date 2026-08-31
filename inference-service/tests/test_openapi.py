"""OpenAPI document."""

from fastapi.testclient import TestClient


def test_openapi_document_is_generated(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["openapi"].startswith("3.")


def test_all_contract_endpoints_are_published(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/health", "/model/info", "/predict"}


def test_future_prediction_response_schema_is_published(client: TestClient) -> None:
    """The success contract is legible now, even though no handler fulfils it yet."""
    document = client.get("/openapi.json").json()

    responses = document["paths"]["/predict"]["post"]["responses"]
    assert "PredictionResponse" in responses["200"]["content"]["application/json"]["schema"]["$ref"]

    prediction = document["components"]["schemas"]["PredictionResponse"]
    assert set(prediction["properties"]) == {"is_anomaly", "anomaly_score", "model_name", "model_version"}
    assert set(prediction["required"]) == {"is_anomaly", "anomaly_score", "model_name", "model_version"}


def test_future_model_info_schema_is_published(client: TestClient) -> None:
    document = client.get("/openapi.json").json()

    responses = document["paths"]["/model/info"]["get"]["responses"]
    assert "ModelInfoResponse" in responses["200"]["content"]["application/json"]["schema"]["$ref"]

    info = document["components"]["schemas"]["ModelInfoResponse"]
    assert {"model_name", "model_version", "feature_order", "canonical_units"} <= set(info["properties"])


def test_request_schema_admits_no_units(client: TestClient) -> None:
    """Nothing in the canonical contract lets a source unit through."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    features = schemas["CanonicalFeatures"]
    assert features["additionalProperties"] is False
    assert "unit" not in features["properties"]
    for definition in features["properties"].values():
        assert definition["type"] == "number"
