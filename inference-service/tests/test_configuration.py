"""Configuration that would make correct operation impossible is rejected."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_the_documented_defaults_are_valid() -> None:
    settings = Settings()

    assert settings.model_name
    assert settings.model_version
    assert str(settings.model_artifact_path)


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"MODEL_NAME": ""}, id="empty-model-name"),
        pytest.param({"MODEL_VERSION": ""}, id="empty-model-version"),
    ],
)
def test_empty_model_expectations_are_rejected(override: dict) -> None:
    """These are checked against the artifact at load time.

    An empty expectation would silently match nothing, turning a deployment
    pointed at the wrong artifact into a confusing runtime failure.
    """
    with pytest.raises(ValidationError):
        Settings(**override)


@pytest.mark.parametrize("path", ["", "   ", "."])
def test_an_empty_artifact_path_is_rejected(path: str) -> None:
    """`Path("")` quietly becomes `.`, which is never an artifact."""
    with pytest.raises(ValidationError):
        Settings(MODEL_ARTIFACT_PATH=path)


def test_a_real_artifact_path_is_accepted() -> None:
    settings = Settings(MODEL_ARTIFACT_PATH="ml/artifacts/model.joblib")

    assert str(settings.model_artifact_path) == "ml/artifacts/model.joblib"


@pytest.mark.parametrize("level", ["DEBUG", "info", "WARNING", "error", "CRITICAL"])
def test_valid_log_levels_are_accepted(level: str) -> None:
    assert Settings(LOG_LEVEL=level).log_level == level


@pytest.mark.parametrize("override", [{"LOG_LEVEL": "LOUD"}, {"LOG_FORMAT": "xml"}])
def test_invalid_logging_configuration_is_rejected(override: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(**override)
