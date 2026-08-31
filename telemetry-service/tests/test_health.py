"""Health endpoint."""

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "telemetry-service", "version": "0.1.0"}


def test_health_reports_nothing_about_dependencies(client: TestClient) -> None:
    """Phase 1 has no database and no inference, so health must claim neither."""
    body = client.get("/health").json()

    assert set(body) == {"status", "service", "version"}


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


class _Client:
    """A Mongo client stand-in whose ping succeeds or fails."""

    def __init__(self, reachable: bool) -> None:
        self.admin = self
        self._reachable = reachable

    async def command(self, name: str):
        if not self._reachable:
            raise RuntimeError("no servers available")
        return {"ok": 1}


def test_readiness_is_ready_when_the_store_answers(app) -> None:
    app.state.mongo_client = _Client(reachable=True)

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["dependencies"] == [
        {"name": "telemetry_store", "ready": True, "detail": None}
    ]


def test_readiness_is_503_when_the_store_is_unreachable(app) -> None:
    """Ingestion is fail-closed on persistence, so this instance is not ready."""
    app.state.mongo_client = _Client(reachable=False)

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["dependencies"][0]["ready"] is False


def test_readiness_is_503_when_no_store_is_configured(app) -> None:
    app.state.mongo_client = None

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"][0]["detail"] == "not configured"


def test_readiness_leaks_no_connection_details(app) -> None:
    app.state.mongo_client = _Client(reachable=False)

    body = TestClient(app).get("/health/ready").text

    for leak in ("mongodb://", "27017", "Traceback", "pymongo"):
        assert leak not in body


def test_readiness_makes_no_claim_about_inference(app) -> None:
    """Inference failure is fail-open, so it must never make this unready."""
    app.state.mongo_client = _Client(reachable=True)

    body = TestClient(app).get("/health/ready").json()

    assert [dependency["name"] for dependency in body["dependencies"]] == ["telemetry_store"]


def test_liveness_stays_up_when_the_store_is_down(app) -> None:
    """A database blip must not look like a dead process."""
    app.state.mongo_client = _Client(reachable=False)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
