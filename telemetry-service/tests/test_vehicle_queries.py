"""Vehicle history endpoints: parameters validated, no records invented."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import error_code

RANGE = {"start": "2026-08-31T00:00:00Z", "end": "2026-09-01T00:00:00Z"}
PATHS = ["/api/v1/vehicles/veh-tw-0142/telemetry", "/api/v1/vehicles/veh-tw-0142/anomalies"]


@pytest.mark.parametrize("path", PATHS)
def test_history_is_refused_rather_than_faked(client: TestClient, path: str) -> None:
    response = client.get(path, params=RANGE)

    assert response.status_code == 501
    assert error_code(response) == "NOT_IMPLEMENTED"


@pytest.mark.parametrize("path", PATHS)
def test_history_never_fabricates_database_records(client: TestClient, path: str) -> None:
    """Not an empty page either: an empty page would claim the store was consulted."""
    body = client.get(path, params=RANGE).json()

    assert set(body) == {"error"}
    assert "items" not in str(body)
    assert "count" not in str(body)


@pytest.mark.parametrize("path", PATHS)
def test_time_range_bounds_must_be_timezone_aware(client: TestClient, path: str) -> None:
    response = client.get(path, params={"start": "2026-08-31T00:00:00", "end": "2026-09-01T00:00:00Z"})

    assert response.status_code == 422
    assert error_code(response) == "NAIVE_TIMESTAMP"


@pytest.mark.parametrize("path", PATHS)
def test_start_must_precede_end(client: TestClient, path: str) -> None:
    response = client.get(path, params={"start": RANGE["end"], "end": RANGE["start"]})

    assert response.status_code == 422
    assert error_code(response) == "INVALID_TIME_RANGE"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("params", [{"start": RANGE["start"]}, {"end": RANGE["end"]}, {}])
def test_both_bounds_are_required(client: TestClient, path: str, params: dict[str, str]) -> None:
    assert client.get(path, params=params).status_code == 422


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_limit_is_bounded(client: TestClient, path: str, limit: int) -> None:
    assert client.get(path, params=RANGE | {"limit": limit}).status_code == 422


@pytest.mark.parametrize("path", PATHS)
def test_offset_bounds_are_accepted(client: TestClient, path: str) -> None:
    params = {"start": "2026-08-31T00:00:00+08:00", "end": "2026-09-01T00:00:00-07:00"}

    assert client.get(path, params=params).status_code != 422
