"""Logging that keeps its context, and request correlation.

The application attaches `event_id`, `vehicle_id` and friends to log records
through `extra=`. Nothing renders them unless a formatter looks for them, so
these tests pin the thing that makes an operational log line useful at all.

Log *prose* is deliberately not asserted — only that the identifiers survive.
"""

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.logging import JsonFormatter, TextFormatter, configure_logging
from app.core.request_id import REQUEST_ID_HEADER, current_request_id
from tests.conftest import TELEMETRY_URL
from tests.fakes import StubInferencePort
from app.domain.inference import InferenceErrorCode


def _record_with(extra: dict[str, Any]) -> logging.LogRecord:
    made = logging.LogRecord(
        name="app.test", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="something happened", args=None, exc_info=None,
    )
    for key, value in extra.items():
        setattr(made, key, value)
    return made


# --------------------------------------------------------------------------- #
# The formatters keep the context
# --------------------------------------------------------------------------- #


def test_json_logs_carry_the_attached_identifiers() -> None:
    line = JsonFormatter().format(
        _record_with({"event_id": "evt-1", "vehicle_id": "veh-1", "error_code": "INFERENCE_TIMEOUT"})
    )

    payload = json.loads(line)
    assert payload["event_id"] == "evt-1"
    assert payload["vehicle_id"] == "veh-1"
    assert payload["error_code"] == "INFERENCE_TIMEOUT"


def test_json_logs_carry_level_logger_and_message() -> None:
    payload = json.loads(JsonFormatter().format(_record_with({})))

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "something happened"
    assert "timestamp" in payload


def test_json_logs_are_one_parseable_object_per_line() -> None:
    line = JsonFormatter().format(_record_with({"event_id": "evt-1"}))

    assert "\n" not in line
    assert isinstance(json.loads(line), dict)


def test_text_logs_carry_the_attached_identifiers() -> None:
    line = TextFormatter().format(_record_with({"event_id": "evt-1", "error_code": "X"}))

    assert "event_id=evt-1" in line
    assert "error_code=X" in line


def test_a_traceback_goes_to_the_log_not_to_a_response() -> None:
    try:
        raise ValueError("internal detail")
    except ValueError:
        import sys

        made = _record_with({})
        made.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(made))

    assert "ValueError" in payload["exception"]


def test_configuring_logging_twice_does_not_duplicate_handlers() -> None:
    configure_logging("INFO", "json")
    configure_logging("INFO", "json")

    assert len(logging.getLogger().handlers) == 1


# --------------------------------------------------------------------------- #
# The identifiers actually reach a log line during ingestion
# --------------------------------------------------------------------------- #


def test_a_fail_open_ingest_logs_the_identifiers(
    app, valid_payload, caplog: pytest.LogCaptureFixture
) -> None:
    from app.api.dependencies import get_inference_port

    app.dependency_overrides[get_inference_port] = lambda: StubInferencePort(
        fail_with=InferenceErrorCode.TIMEOUT
    )

    with caplog.at_level(logging.WARNING):
        TestClient(app).post(TELEMETRY_URL, json=valid_payload)

    entry = next(r for r in caplog.records if "without a verdict" in r.getMessage())
    assert entry.event_id == valid_payload["event_id"]
    assert entry.vehicle_id == valid_payload["vehicle_id"]
    assert entry.site_id == valid_payload["site_id"]
    assert entry.error_code == "INFERENCE_TIMEOUT"


# --------------------------------------------------------------------------- #
# Request correlation
# --------------------------------------------------------------------------- #


def test_a_request_id_is_generated_when_none_is_supplied(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers[REQUEST_ID_HEADER]) >= 8


def test_a_supplied_request_id_is_echoed_back(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"


def test_each_request_gets_its_own_generated_id(client: TestClient) -> None:
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    assert first != second


def test_an_oversized_request_id_is_bounded(client: TestClient) -> None:
    """A caller-supplied value is echoed, so it is not trusted as-is."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "x" * 5000})

    assert len(response.headers[REQUEST_ID_HEADER]) <= 128


def test_error_responses_also_carry_a_request_id(client: TestClient) -> None:
    response = client.post(TELEMETRY_URL, json={})

    assert response.status_code == 422
    assert response.headers[REQUEST_ID_HEADER]


def test_the_request_id_does_not_leak_between_requests(client: TestClient) -> None:
    client.get("/health", headers={REQUEST_ID_HEADER: "first"})

    assert current_request_id() == ""


def test_the_request_id_is_forwarded_to_the_inference_service(
    client: TestClient, valid_payload, inference: StubInferencePort
) -> None:
    """One id ties an ingest request to the prediction it triggered."""
    import httpx

    from app.infrastructure.http_inference_client import HttpInferenceClient
    from app.core.request_id import _request_id

    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(REQUEST_ID_HEADER))
        return httpx.Response(
            200,
            json={
                "is_anomaly": False,
                "anomaly_score": -0.1,
                "model_name": "isolation-forest-telemetry",
                "model_version": "0.1.0",
            },
        )

    async def run() -> None:
        token = _request_id.set("trace-xyz")
        async with httpx.AsyncClient(
            base_url="http://inference.internal", transport=httpx.MockTransport(handler)
        ) as transport_client:
            from app.domain.units import MetricName

            await HttpInferenceClient(transport_client).predict(
                {metric: 1.0 for metric in MetricName}
            )
        _request_id.reset(token)

    import anyio

    anyio.run(run)

    assert seen == ["trace-xyz"]
