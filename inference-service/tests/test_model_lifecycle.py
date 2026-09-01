"""The model is loaded once per process, not once per request."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import NORMAL_SAMPLE, PREDICT_URL


def test_the_artifact_is_read_from_disk_exactly_once(
    artifact_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup loads it; requests reuse it."""
    import app.main as main_module
    from app.core.config import Settings

    loads: list[Path] = []
    original = main_module.load_artifact

    def counting_load(path, **kwargs):
        loads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(main_module, "load_artifact", counting_load)
    monkeypatch.setattr(main_module, "get_settings", lambda: Settings(MODEL_ARTIFACT_PATH=artifact_path))

    with TestClient(main_module.create_app()) as client:
        for _ in range(5):
            assert client.post(PREDICT_URL, json={"features": dict(NORMAL_SAMPLE)}).status_code == 200
        assert client.get("/model/info").status_code == 200

    assert len(loads) == 1, f"artifact was loaded {len(loads)} times"


def test_every_request_is_served_by_the_same_model_object(client: TestClient) -> None:
    identities = set()

    for _ in range(3):
        client.post(PREDICT_URL, json={"features": dict(NORMAL_SAMPLE)})
        identities.add(id(client.app.state.inference_service))

    assert len(identities) == 1


def test_repeated_requests_return_identical_results(client: TestClient) -> None:
    """No per-request state, so the same input always scores the same."""
    body = {"features": dict(NORMAL_SAMPLE)}

    responses = [client.post(PREDICT_URL, json=body).json() for _ in range(4)]

    assert all(response == responses[0] for response in responses)


def test_no_training_happens_at_startup(
    artifact_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serving never trains. If it did, startup would depend on the generator."""
    import ml.train as train_module
    from app.core.config import Settings
    import app.main as main_module

    def explode(*args, **kwargs):
        raise AssertionError("the service trained a model at startup")

    monkeypatch.setattr(train_module, "train", explode)
    monkeypatch.setattr(main_module, "get_settings", lambda: Settings(MODEL_ARTIFACT_PATH=artifact_path))

    with TestClient(main_module.create_app()) as client:
        assert client.get("/health").json()["model_loaded"] is True
