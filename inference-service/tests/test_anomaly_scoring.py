"""The anomaly_score transformation and the verdict it accompanies.

The public score is `-decision_function(x)`, so higher means more anomalous and
the model's own decision boundary lands on zero. These tests pin that
orientation, because the sklearn-native one is the opposite way round and an
accidental sign flip would be invisible in every other test.
"""

import math

import pytest

from app.domain.features import FEATURE_ORDER
from app.domain.prediction import Prediction
from ml.generate_training_data import generate_samples
from tests.conftest import ANOMALOUS_SAMPLE, NORMAL_SAMPLE


def vector(sample: dict[str, float]) -> list[list[float]]:
    return [[sample[name] for name in FEATURE_ORDER]]


def test_the_score_is_the_negated_decision_function(model) -> None:
    """The exact transformation, pinned."""
    raw = float(model._estimator.decision_function(vector(NORMAL_SAMPLE))[0])

    assert model.predict(NORMAL_SAMPLE).anomaly_score == pytest.approx(-raw)


def test_higher_means_more_anomalous(model) -> None:
    normal = model.predict(NORMAL_SAMPLE)
    extreme = model.predict(ANOMALOUS_SAMPLE)

    assert extreme.anomaly_score > normal.anomaly_score


def test_a_normal_sample_scores_at_or_below_zero(model) -> None:
    assert model.predict(NORMAL_SAMPLE).anomaly_score <= 0


def test_an_extreme_sample_scores_above_zero(model) -> None:
    assert model.predict(ANOMALOUS_SAMPLE).anomaly_score > 0


@pytest.fixture(scope="module")
def scored_population(model) -> list[Prediction]:
    """A population scored once, spanning both sides of the decision boundary."""
    samples = generate_samples(count=300, seed=4242)
    samples.extend([dict(NORMAL_SAMPLE), dict(ANOMALOUS_SAMPLE)])
    return [model.predict(sample) for sample in samples]


def test_the_verdict_and_the_score_agree_about_the_boundary(
    scored_population: list[Prediction],
) -> None:
    """`is_anomaly` and `anomaly_score > 0` must never disagree.

    The verdict comes from IsolationForest's own `predict`; the score is derived
    separately from its decision function. This is what keeps the two in step.
    """
    for prediction in scored_population:
        assert prediction.is_anomaly == (prediction.anomaly_score > 0), prediction


def test_the_population_straddles_the_boundary(scored_population: list[Prediction]) -> None:
    """Otherwise the agreement test above would be vacuous."""
    assert any(prediction.is_anomaly for prediction in scored_population)
    assert any(not prediction.is_anomaly for prediction in scored_population)


def test_every_anomaly_outranks_every_normal_sample(
    scored_population: list[Prediction],
) -> None:
    """The two sides of the boundary do not interleave."""
    anomalous = [p.anomaly_score for p in scored_population if p.is_anomaly]
    normal = [p.anomaly_score for p in scored_population if not p.is_anomaly]

    assert min(anomalous) > max(normal)
    assert max(normal) <= 0 < min(anomalous)


def test_the_score_is_finite_for_every_sample(scored_population: list[Prediction]) -> None:
    for prediction in scored_population:
        assert math.isfinite(prediction.anomaly_score)


def test_scoring_is_deterministic(model) -> None:
    """The same vector always scores the same. No randomness at predict time."""
    first = model.predict(NORMAL_SAMPLE)
    second = model.predict(NORMAL_SAMPLE)

    assert first == second


def test_the_score_is_not_bounded_to_zero_and_one(
    scored_population: list[Prediction],
) -> None:
    """It is a ranking score, not a probability, and is not squashed."""
    scores = [prediction.anomaly_score for prediction in scored_population]

    assert min(scores) < 0, "a probability could never be negative"


def test_the_prediction_carries_the_model_identity(model) -> None:
    prediction = model.predict(NORMAL_SAMPLE)

    assert isinstance(prediction, Prediction)
    assert prediction.model_name == model.metadata.model_name
    assert prediction.model_version == model.metadata.model_version


def test_ranking_orders_samples_by_how_unusual_they_are(model) -> None:
    """Progressively more extreme samples score progressively higher."""
    ladder = [
        {**NORMAL_SAMPLE},
        {**NORMAL_SAMPLE, "battery_temperature": 60.0},
        {**NORMAL_SAMPLE, "battery_temperature": 95.0, "battery_current": -600.0},
        dict(ANOMALOUS_SAMPLE),
    ]

    scores = [model.predict(sample).anomaly_score for sample in ladder]

    assert scores == sorted(scores), scores


def test_a_missing_feature_is_an_error_not_a_guess(model) -> None:
    with pytest.raises(KeyError):
        model.predict({name: 1.0 for name in FEATURE_ORDER if name != "speed"})
