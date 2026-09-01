"""One authoritative feature order, used everywhere.

Column order is the kind of thing that breaks silently: a model keeps returning
plausible numbers while scoring the wrong quantities. These tests exist so that
cannot happen unnoticed.
"""

from app.api import schemas
from app.domain.features import CANONICAL_UNITS, FEATURE_ORDER
from ml.generate_training_data import generate_samples, to_matrix

EXPECTED_ORDER = (
    "soc",
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "speed",
    "motor_rpm",
)
"""Written out literally. If FEATURE_ORDER changes, this must be changed too —
which is the point: it is a model version change, not a refactor."""


def test_the_authoritative_order_is_exact() -> None:
    assert FEATURE_ORDER == EXPECTED_ORDER


def test_the_order_has_no_duplicates() -> None:
    assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER)


def test_every_feature_has_a_canonical_unit() -> None:
    assert tuple(CANONICAL_UNITS) == FEATURE_ORDER


def test_the_request_schema_covers_exactly_the_ordered_features() -> None:
    assert set(schemas.CanonicalFeatures.model_fields) == set(FEATURE_ORDER)


def test_training_lays_out_columns_in_the_authoritative_order() -> None:
    sample = generate_samples(count=1, seed=1)[0]

    row = to_matrix([sample], FEATURE_ORDER)[0]

    assert row == [sample[name] for name in EXPECTED_ORDER]


def test_the_artifact_records_the_order_it_was_trained_with(loaded_artifact) -> None:
    assert loaded_artifact.metadata.feature_order == EXPECTED_ORDER


def test_inference_orders_columns_the_same_way_as_training(model) -> None:
    """The model reads the vector by name, in its own recorded order."""
    assert model.metadata.feature_order == EXPECTED_ORDER


def test_the_published_metadata_exposes_the_same_order(client) -> None:
    assert client.get("/model/info").json()["feature_order"] == list(EXPECTED_ORDER)


def test_client_key_order_cannot_reorder_the_vector(model) -> None:
    """Two payloads differing only in key order must score identically."""
    forward = {name: float(index + 1) for index, name in enumerate(FEATURE_ORDER)}
    backward = dict(reversed(list(forward.items())))

    assert list(forward) != list(backward)
    assert model.predict(forward) == model.predict(backward)


def test_a_permuted_vector_scores_differently(model) -> None:
    """Proof the order is load-bearing: same numbers, different columns.

    If this ever passed, the ordering guarantee above would be vacuous.
    """
    straight = {name: float(index + 1) for index, name in enumerate(FEATURE_ORDER)}
    permuted = dict(zip(FEATURE_ORDER, reversed(list(straight.values()))))

    assert model.predict(straight).anomaly_score != model.predict(permuted).anomaly_score
