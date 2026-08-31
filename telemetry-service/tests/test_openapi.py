"""OpenAPI document."""

from fastapi.testclient import TestClient


def test_openapi_document_is_generated(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["openapi"].startswith("3.")


def test_all_contract_endpoints_are_published(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {
        "/health",
        "/health/ready",
        "/api/v1/telemetry",
        "/api/v1/vehicles/{vehicle_id}/telemetry",
        "/api/v1/vehicles/{vehicle_id}/anomalies",
    }


def test_ingest_publishes_every_documented_outcome(client: TestClient) -> None:
    responses = client.get("/openapi.json").json()["paths"]["/api/v1/telemetry"]["post"]["responses"]

    assert "TelemetryEventResponse" in responses["201"]["content"]["application/json"]["schema"]["$ref"]
    assert "TelemetryEventResponse" in responses["200"]["content"]["application/json"]["schema"]["$ref"]
    assert "ErrorEnvelope" in responses["422"]["content"]["application/json"]["schema"]["$ref"]
    assert "ErrorEnvelope" in responses["503"]["content"]["application/json"]["schema"]["$ref"]
    assert "501" not in responses


def test_inference_status_publishes_the_unscored_state(client: TestClient) -> None:
    """PENDING is part of the contract: stored, but carrying no verdict."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert schemas["InferenceOutcome"]["properties"]["status"]["enum"] == [
        "PENDING",
        "COMPLETED",
        "FAILED",
    ]


def test_request_schema_documents_units_per_metric(client: TestClient) -> None:
    """Each metric advertises its own accepted units, not a request-level system."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert schemas["SpeedMeasurement"]["properties"]["unit"]["enum"] == ["km/h", "mph", "m/s"]
    assert schemas["BatteryTemperatureMeasurement"]["properties"]["unit"]["enum"] == ["degC", "degF", "K"]
    assert "unit_system" not in schemas["TelemetryIngestRequest"]["properties"]
