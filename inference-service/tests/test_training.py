"""The training script and the artifact it produces."""

from datetime import datetime, timezone

import joblib
import pytest
from sklearn.ensemble import IsolationForest

from app.domain.features import CANONICAL_UNITS, FEATURE_ORDER
from app.infrastructure.artifact import ARTIFACT_SCHEMA_VERSION, REQUIRED_KEYS
from ml.train import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    HYPERPARAMETERS,
    train,
)
from tests.conftest import ANOMALOUS_SAMPLE, FIXED_TRAINED_AT, NORMAL_SAMPLE

SMALL = {"sample_count": 1_200, "trained_at": FIXED_TRAINED_AT}


def test_training_produces_a_fitted_isolation_forest() -> None:
    artifact = train(**SMALL)

    model = artifact["model"]
    assert isinstance(model, IsolationForest)
    assert hasattr(model, "estimators_"), "model is not fitted"


def test_the_artifact_carries_every_required_key() -> None:
    assert REQUIRED_KEYS <= train(**SMALL).keys()


def test_the_artifact_records_its_identity() -> None:
    artifact = train(**SMALL)

    assert artifact["model_name"] == DEFAULT_MODEL_NAME
    assert artifact["model_version"] == DEFAULT_MODEL_VERSION
    assert artifact["algorithm"] == "sklearn.ensemble.IsolationForest"
    assert artifact["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_the_artifact_records_the_feature_vocabulary() -> None:
    artifact = train(**SMALL)

    assert tuple(artifact["feature_order"]) == FEATURE_ORDER
    assert artifact["canonical_units"] == dict(CANONICAL_UNITS)


def test_the_artifact_records_how_it_was_trained() -> None:
    """Enough to reproduce the run from the artifact alone."""
    training = train(**SMALL)["training"]

    assert training["sample_count"] == 1_200
    assert training["synthetic"] is True
    assert training["hyperparameters"] == HYPERPARAMETERS
    assert "data_seed" in training


def test_hyperparameters_are_explicit_not_defaults() -> None:
    """The model is reproducible from the training module alone."""
    for name in ("n_estimators", "max_samples", "contamination", "random_state"):
        assert name in HYPERPARAMETERS


def test_the_random_state_is_fixed() -> None:
    assert HYPERPARAMETERS["random_state"] is not None


def test_training_is_deterministic() -> None:
    """Same seed, same data, same scores."""
    first = train(**SMALL)["model"]
    second = train(**SMALL)["model"]

    vectors = [
        [NORMAL_SAMPLE[name] for name in FEATURE_ORDER],
        [ANOMALOUS_SAMPLE[name] for name in FEATURE_ORDER],
    ]
    assert list(first.decision_function(vectors)) == list(second.decision_function(vectors))


def test_a_pinned_timestamp_makes_the_artifact_reproducible() -> None:
    assert train(**SMALL)["trained_at"] == FIXED_TRAINED_AT


def test_trained_at_defaults_to_now_and_is_utc() -> None:
    trained_at = train(sample_count=1_200)["trained_at"]

    assert trained_at.tzinfo is not None
    assert trained_at <= datetime.now(timezone.utc)


def test_the_artifact_round_trips_through_joblib(tmp_path) -> None:
    path = tmp_path / "model.joblib"
    artifact = train(**SMALL)

    joblib.dump(artifact, path)
    restored = joblib.load(path)

    vector = [[NORMAL_SAMPLE[name] for name in FEATURE_ORDER]]
    assert restored["model_name"] == artifact["model_name"]
    assert list(restored["model"].decision_function(vector)) == list(
        artifact["model"].decision_function(vector)
    )


def test_the_trained_model_separates_normal_from_extreme() -> None:
    """The point of training: the two fixtures land on opposite sides.

    This is a property of these deliberately-constructed fixtures, not a claim
    that every synthetic point receives a particular label.
    """
    model = train(**SMALL)["model"]

    normal = [[NORMAL_SAMPLE[name] for name in FEATURE_ORDER]]
    extreme = [[ANOMALOUS_SAMPLE[name] for name in FEATURE_ORDER]]

    assert int(model.predict(normal)[0]) == 1
    assert int(model.predict(extreme)[0]) == -1


def test_contamination_bounds_the_false_positive_rate() -> None:
    """Held-out normal data is flagged at roughly the configured rate."""
    from ml.generate_training_data import generate_samples, to_matrix

    model = train(**SMALL)["model"]
    held_out = to_matrix(generate_samples(count=1_000, seed=99), FEATURE_ORDER)

    flagged = sum(1 for label in model.predict(held_out) if label == -1)

    assert flagged / 1_000 == pytest.approx(HYPERPARAMETERS["contamination"], abs=0.03)
