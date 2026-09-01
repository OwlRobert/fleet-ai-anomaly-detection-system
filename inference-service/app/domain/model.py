"""The model, as the application needs it."""

from typing import Mapping, Protocol

from app.domain.prediction import ModelMetadata, Prediction


class AnomalyModel(Protocol):
    """A loaded model that can score one canonical feature vector."""

    @property
    def metadata(self) -> ModelMetadata:
        """What this model is, for `/model/info`."""
        ...

    def predict(self, features: Mapping[str, float]) -> Prediction:
        """Score one vector of canonical features.

        Args:
            features: Canonical values keyed by feature name. Key order is
                irrelevant; the model orders the columns itself.
        """
        ...
