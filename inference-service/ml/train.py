"""Train the anomaly model and write the artifact.

Training is a separate concern from serving: this script runs by hand (or in a
build), writes one joblib file, and exits. The service never trains — it only
loads what this produced.

Why IsolationForest: anomaly labels for fleet telemetry are scarce, and this
project is about demonstrating a complete model lifecycle rather than maximizing
predictive accuracy. IsolationForest needs no labels, trains in seconds, and has
an interpretable decision boundary. It is a reasonable default for unsupervised
outlier detection, not a claim of production-grade accuracy.

Run it with:

    cd inference-service
    PYTHONPATH=. python ml/train.py

which writes ``ml/artifacts/isolation_forest_v0_1_0.joblib``, the path
``.env.example`` documents. The artifact is git-ignored and rebuilt from here.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import IsolationForest

from app.domain.features import FEATURE_ORDER
from app.infrastructure.artifact import build_artifact
from ml.generate_training_data import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    generate_samples,
    to_matrix,
)

DEFAULT_MODEL_NAME = "isolation-forest-telemetry"
DEFAULT_MODEL_VERSION = "0.1.0"
DEFAULT_ARTIFACT_PATH = Path("ml/artifacts/isolation_forest_v0_1_0.joblib")

# Stated explicitly rather than left to defaults, so the trained model is
# reproducible from this file alone.
HYPERPARAMETERS: dict[str, Any] = {
    # More trees than the default 100: cheap here, and it steadies the score.
    "n_estimators": 400,
    # Rows each tree sees. The IsolationForest paper's default of 256 detects
    # points outside the data envelope, but is too coarse to resolve the *empty
    # interior regions* this dataset has - 0 km/h at 9000 rpm is impossible, yet
    # both values are individually ordinary. Larger subsamples build deeper
    # trees that carve those holes out. Fixed rather than "auto" so the model
    # does not change shape when the dataset size changes.
    "max_samples": 1024,
    # The share of training data treated as outliers. This is what places the
    # decision boundary, so it is the one hyperparameter with a directly visible
    # effect on `is_anomaly`. 2% is a demonstration assumption.
    "contamination": 0.02,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
}


def train(
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    trained_at: datetime | None = None,
) -> dict[str, Any]:
    """Generate data, fit the model, and assemble the artifact.

    Args:
        sample_count: How many synthetic samples to train on.
        seed: Seed for the data generator.
        model_name: Name recorded in the artifact and served by `/model/info`.
        model_version: Version recorded in the artifact.
        trained_at: Timestamp recorded in the artifact. Pass a fixed value to
            get a byte-identical artifact from an identical run; the default is
            the current time.

    Returns:
        The artifact dict, ready to serialize.
    """
    samples = generate_samples(count=sample_count, seed=seed)
    matrix = to_matrix(samples, FEATURE_ORDER)

    model = IsolationForest(**HYPERPARAMETERS)
    model.fit(matrix)

    return build_artifact(
        model=model,
        model_name=model_name,
        model_version=model_version,
        trained_at=trained_at or datetime.now(timezone.utc),
        training={
            "sample_count": sample_count,
            "data_seed": seed,
            "hyperparameters": dict(HYPERPARAMETERS),
            "synthetic": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    arguments = parser.parse_args()

    artifact = train(
        sample_count=arguments.samples,
        seed=arguments.seed,
        model_name=arguments.model_name,
        model_version=arguments.model_version,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, arguments.output)

    print(f"wrote {arguments.output}")
    print(f"  model        {artifact['model_name']} {artifact['model_version']}")
    print(f"  algorithm    {artifact['algorithm']}")
    print(f"  features     {', '.join(artifact['feature_order'])}")
    print(f"  trained on   {arguments.samples} synthetic samples, seed {arguments.seed}")
    print(f"  sklearn      {artifact['sklearn_version']}")


if __name__ == "__main__":
    main()
