"""The IsolationForest behind the ``AnomalyModel`` interface.

This is the only module that knows scikit-learn's scoring conventions, and it
translates them once so nothing above it has to. The API never sees a raw
``decision_function`` value.
"""

from typing import Mapping

from app.domain.prediction import ModelMetadata, Prediction
from app.infrastructure.artifact import LoadedArtifact

OUTLIER_LABEL = -1
"""What IsolationForest returns for a sample it considers an outlier."""


class IsolationForestAnomalyModel:
    """Scores canonical feature vectors with a fitted IsolationForest."""

    def __init__(self, artifact: LoadedArtifact) -> None:
        self._estimator = artifact.model
        self._metadata = artifact.metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, features: Mapping[str, float]) -> Prediction:
        """Score one vector of canonical features.

        The verdict comes from the estimator's own ``predict``, and the score is
        its decision function **negated**, so that higher means more anomalous
        and the decision boundary lands on zero.

        Args:
            features: Canonical values keyed by feature name.

        Returns:
            The prediction, carrying the identity of the model that made it.

        Raises:
            KeyError: If a feature the model needs is absent. The API schema
                rejects such a payload long before this point.
        """
        vector = [float(features[name]) for name in self._metadata.feature_order]

        label = int(self._estimator.predict([vector])[0])
        decision = float(self._estimator.decision_function([vector])[0])

        return Prediction(
            is_anomaly=label == OUTLIER_LABEL,
            anomaly_score=-decision,
            model_name=self._metadata.model_name,
            model_version=self._metadata.model_version,
        )
