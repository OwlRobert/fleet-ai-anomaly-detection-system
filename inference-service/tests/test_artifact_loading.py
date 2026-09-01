"""Loading the artifact, and refusing to load a bad one.

Every rejection here is a case where serving *something* would be worse than
serving nothing: a vector scored against the wrong columns still returns a
plausible-looking number.
"""

from pathlib import Path

import joblib
import pytest

from app.core.errors import ArtifactLoadError
from app.domain.features import CANONICAL_UNITS, FEATURE_ORDER
from app.infrastructure.artifact import ARTIFACT_SCHEMA_VERSION, load_artifact
from ml.train import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, train
from tests.conftest import FIXED_TRAINED_AT


@pytest.fixture(scope="module")
def _trained_once() -> dict:
    """Train once for the module; the tests below only mutate top-level keys."""
    return train(sample_count=1_200, trained_at=FIXED_TRAINED_AT)


@pytest.fixture
def artifact(_trained_once: dict) -> dict:
    """A fresh top-level copy, so one test's mutation cannot leak into another."""
    return dict(_trained_once)


def write(path: Path, payload: object) -> Path:
    joblib.dump(payload, path)
    return path


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_a_valid_artifact_loads(loaded_artifact) -> None:
    assert loaded_artifact.metadata.model_name == DEFAULT_MODEL_NAME
    assert loaded_artifact.metadata.model_version == DEFAULT_MODEL_VERSION
    assert loaded_artifact.metadata.algorithm == "sklearn.ensemble.IsolationForest"


def test_loading_records_the_digest_of_the_file(loaded_artifact) -> None:
    digest = loaded_artifact.metadata.artifact_sha256

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_the_digest_identifies_the_file(tmp_path: Path, artifact: dict) -> None:
    first = load_artifact(write(tmp_path / "a.joblib", artifact))
    same = load_artifact(write(tmp_path / "b.joblib", artifact))

    assert first.metadata.artifact_sha256 == same.metadata.artifact_sha256


def test_loading_preserves_the_training_provenance(loaded_artifact) -> None:
    assert loaded_artifact.training["synthetic"] is True


def test_the_configured_name_and_version_are_accepted_when_they_match(
    tmp_path: Path, artifact: dict
) -> None:
    path = write(tmp_path / "model.joblib", artifact)

    loaded = load_artifact(
        path, expected_name=DEFAULT_MODEL_NAME, expected_version=DEFAULT_MODEL_VERSION
    )

    assert loaded.metadata.model_name == DEFAULT_MODEL_NAME


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_missing_artifact_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactLoadError, match="no model artifact"):
        load_artifact(tmp_path / "absent.joblib")


def test_a_directory_is_not_an_artifact(tmp_path: Path) -> None:
    with pytest.raises(ArtifactLoadError):
        load_artifact(tmp_path)


def test_a_corrupt_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"this is not a joblib file at all")

    with pytest.raises(ArtifactLoadError, match="could not be read"):
        load_artifact(path)


def test_a_truncated_artifact_is_refused(tmp_path: Path, artifact: dict) -> None:
    path = write(tmp_path / "model.joblib", artifact)
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])

    with pytest.raises(ArtifactLoadError):
        load_artifact(path)


def test_something_that_is_not_an_artifact_dict_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactLoadError, match="not an artifact dict"):
        load_artifact(write(tmp_path / "model.joblib", ["not", "an", "artifact"]))


def test_an_artifact_missing_keys_is_refused(tmp_path: Path, artifact: dict) -> None:
    del artifact["feature_order"]
    del artifact["sklearn_version"]

    with pytest.raises(ArtifactLoadError, match="missing keys"):
        load_artifact(write(tmp_path / "model.joblib", artifact))


def test_an_unsupported_schema_version_is_refused(tmp_path: Path, artifact: dict) -> None:
    artifact["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION + 1

    with pytest.raises(ArtifactLoadError, match="schema"):
        load_artifact(write(tmp_path / "model.joblib", artifact))


def test_a_permuted_feature_order_is_refused(tmp_path: Path, artifact: dict) -> None:
    """The same six names in a different order is a different model."""
    artifact["feature_order"] = list(reversed(FEATURE_ORDER))

    with pytest.raises(ArtifactLoadError, match="feature order"):
        load_artifact(write(tmp_path / "model.joblib", artifact))


def test_a_different_feature_set_is_refused(tmp_path: Path, artifact: dict) -> None:
    artifact["feature_order"] = [*FEATURE_ORDER, "tyre_pressure"]

    with pytest.raises(ArtifactLoadError, match="feature order"):
        load_artifact(write(tmp_path / "model.joblib", artifact))


def test_mismatched_canonical_units_are_refused(tmp_path: Path, artifact: dict) -> None:
    """An artifact trained on mph must never score km/h."""
    artifact["canonical_units"] = {**CANONICAL_UNITS, "speed": "mph"}

    with pytest.raises(ArtifactLoadError, match="canonical units"):
        load_artifact(write(tmp_path / "model.joblib", artifact))


def test_an_artifact_without_a_usable_estimator_is_refused(tmp_path: Path, artifact: dict) -> None:
    artifact["model"] = {"not": "an estimator"}

    with pytest.raises(ArtifactLoadError, match="usable estimator"):
        load_artifact(write(tmp_path / "model.joblib", artifact))


def test_a_wrong_model_name_is_refused(tmp_path: Path, artifact: dict) -> None:
    """Catches a deployment pointed at the wrong artifact."""
    path = write(tmp_path / "model.joblib", artifact)

    with pytest.raises(ArtifactLoadError, match="was configured"):
        load_artifact(path, expected_name="some-other-model")


def test_a_wrong_model_version_is_refused(tmp_path: Path, artifact: dict) -> None:
    path = write(tmp_path / "model.joblib", artifact)

    with pytest.raises(ArtifactLoadError, match="version"):
        load_artifact(path, expected_version="9.9.9")


def test_a_refusal_never_raises_a_raw_driver_exception(tmp_path: Path) -> None:
    """Every failure arrives as one application error type."""
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ArtifactLoadError):
        load_artifact(path)


# --------------------------------------------------------------------------- #
# Startup wiring
# --------------------------------------------------------------------------- #


def test_a_failed_load_yields_a_service_without_a_model(tmp_path: Path) -> None:
    from app.core.config import Settings
    from app.main import build_inference_service

    settings = Settings(MODEL_ARTIFACT_PATH=tmp_path / "absent.joblib")

    service = build_inference_service(settings)

    assert service.is_model_loaded is False


def test_a_successful_load_yields_a_service_with_a_model(artifact_path: Path) -> None:
    from app.core.config import Settings
    from app.main import build_inference_service

    settings = Settings(MODEL_ARTIFACT_PATH=artifact_path)

    service = build_inference_service(settings)

    assert service.is_model_loaded is True
    assert service.describe_model().model_name == DEFAULT_MODEL_NAME


def test_startup_does_not_crash_on_a_bad_artifact(tmp_path: Path) -> None:
    """A bad artifact degrades the instance; it does not take the process down."""
    from app.core.config import Settings
    from app.main import build_inference_service

    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a model")

    assert build_inference_service(Settings(MODEL_ARTIFACT_PATH=path)).is_model_loaded is False
