"""OpenAPI document."""

from fastapi.testclient import TestClient


def test_openapi_document_is_generated(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["openapi"].startswith("3.")


def test_all_contract_endpoints_are_published(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/health", "/model/info", "/predict"}


def test_prediction_response_schema_is_published(client: TestClient) -> None:
    """The success contract, now actually fulfilled."""
    document = client.get("/openapi.json").json()

    responses = document["paths"]["/predict"]["post"]["responses"]
    assert "PredictionResponse" in responses["200"]["content"]["application/json"]["schema"]["$ref"]

    prediction = document["components"]["schemas"]["PredictionResponse"]
    assert set(prediction["properties"]) == {"is_anomaly", "anomaly_score", "model_name", "model_version"}
    assert set(prediction["required"]) == {"is_anomaly", "anomaly_score", "model_name", "model_version"}


def test_model_info_schema_is_published(client: TestClient) -> None:
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


def test_the_score_field_is_named_anomaly_score_everywhere(client: TestClient) -> None:
    """A bare `score` would be the old, retired name."""
    document = client.get("/openapi.json").json()

    prediction = document["components"]["schemas"]["PredictionResponse"]["properties"]
    assert "anomaly_score" in prediction
    assert "score" not in prediction


def test_the_score_description_states_its_orientation(client: TestClient) -> None:
    """Higher = more anomalous is the whole contract; it must be written down."""
    document = client.get("/openapi.json").json()

    description = document["components"]["schemas"]["PredictionResponse"]["properties"][
        "anomaly_score"
    ]["description"]

    assert "higher means more anomalous" in description.lower()
    assert "not a probability" in description.lower()


def test_the_model_unavailable_response_is_documented(client: TestClient) -> None:
    document = client.get("/openapi.json").json()

    for path, method in (("/predict", "post"), ("/model/info", "get")):
        responses = document["paths"][path][method]["responses"]
        assert "503" in responses, path
        assert "ErrorEnvelope" in responses["503"]["content"]["application/json"]["schema"]["$ref"]


def test_no_endpoint_still_advertises_not_implemented(client: TestClient) -> None:
    document = client.get("/openapi.json").json()

    for path, operations in document["paths"].items():
        for method, operation in operations.items():
            assert "501" not in operation["responses"], (path, method)


def test_health_publishes_model_loaded(client: TestClient) -> None:
    document = client.get("/openapi.json").json()

    assert "model_loaded" in document["components"]["schemas"]["HealthResponse"]["properties"]


def test_every_operation_keeps_a_summary_and_description(client: TestClient) -> None:
    document = client.get("/openapi.json").json()

    for path, operations in document["paths"].items():
        for method, operation in operations.items():
            assert operation.get("summary"), (path, method)
            assert operation.get("description"), (path, method)
