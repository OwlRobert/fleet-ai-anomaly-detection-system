"""What the model returns, and what describes the model that returned it."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Prediction:
    """One scored feature vector.

    ``anomaly_score`` is **anomaly-oriented**: higher means more anomalous. It
    is the negated IsolationForest decision function, so the model's own
    decision boundary sits at zero — above zero is an outlier, zero or below an
    inlier.

    It is a ranking score, **not a probability**. It is unbounded and is
    deliberately not squashed into ``[0, 1]``, because a bounded number invites
    being read as a likelihood it is not.
    """

    is_anomaly: bool
    anomaly_score: float
    model_name: str
    model_version: str


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Everything `/model/info` publishes about the loaded artifact.

    ``feature_order`` and ``canonical_units`` are contract, not documentation:
    they let a caller verify it is speaking the same dialect as the model.
    """

    model_name: str
    model_version: str
    algorithm: str
    trained_at: datetime
    feature_order: tuple[str, ...]
    canonical_units: Mapping[str, str]
    artifact_sha256: str
    sklearn_version: str
