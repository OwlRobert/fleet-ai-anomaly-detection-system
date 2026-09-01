"""The application boundary between the API and the model.

Its one job beyond delegating is to own the *absence* of a model. Loading
happens once at startup and can fail; every route needs the same answer when it
has, and that answer is an explicit refusal rather than a fabricated verdict.
"""

from typing import Mapping

from app.core.errors import ModelNotLoadedError
from app.domain.model import AnomalyModel
from app.domain.prediction import ModelMetadata, Prediction


class InferenceService:
    """Serves predictions from the model loaded at startup, if there is one."""

    def __init__(self, model: AnomalyModel | None) -> None:
        self._model = model

    @property
    def is_model_loaded(self) -> bool:
        """Whether this process actually has a model to serve."""
        return self._model is not None

    def predict(self, features: Mapping[str, float]) -> Prediction:
        """Score one canonical feature vector.

        Raises:
            ModelNotLoadedError: If no artifact was loaded at startup.
        """
        return self._require_model().predict(features)

    def describe_model(self) -> ModelMetadata:
        """Describe the loaded model.

        Raises:
            ModelNotLoadedError: If no artifact was loaded at startup.
        """
        return self._require_model().metadata

    def _require_model(self) -> AnomalyModel:
        if self._model is None:
            raise ModelNotLoadedError("no model artifact is loaded")
        return self._model
