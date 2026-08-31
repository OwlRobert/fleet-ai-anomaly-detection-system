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
        "/api/v1/telemetry",
        "/api/v1/vehicles/{vehicle_id}/telemetry",
        "/api/v1/vehicles/{vehicle_id}/anomalies",
    }


def test_ingest_publishes_both_the_future_success_and_the_phase_1_refusal(client: TestClient) -> None:
    responses = client.get("/openapi.json").json()["paths"]["/api/v1/telemetry"]["post"]["responses"]

    assert "TelemetryEventResponse" in responses["201"]["content"]["application/json"]["schema"]["$ref"]
    assert "ErrorEnvelope" in responses["422"]["content"]["application/json"]["schema"]["$ref"]
    assert "ErrorEnvelope" in responses["501"]["content"]["application/json"]["schema"]["$ref"]


def test_request_schema_documents_units_per_metric(client: TestClient) -> None:
    """Each metric advertises its own accepted units, not a request-level system."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert schemas["SpeedMeasurement"]["properties"]["unit"]["enum"] == ["km/h", "mph", "m/s"]
    assert schemas["BatteryTemperatureMeasurement"]["properties"]["unit"]["enum"] == ["degC", "degF", "K"]
    assert "unit_system" not in schemas["TelemetryIngestRequest"]["properties"]
