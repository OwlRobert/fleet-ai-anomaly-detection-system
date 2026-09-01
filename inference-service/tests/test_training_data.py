"""The synthetic training-data generator.

The ranges asserted here are the generator's own demonstration assumptions, not
validated vehicle specifications.
"""

import math

import pytest

from app.domain.features import FEATURE_ORDER
from ml.generate_training_data import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DRIVE_RATIO_RPM_PER_KPH,
    generate_samples,
    iter_feature_values,
    to_matrix,
)


def test_the_same_seed_produces_the_same_data() -> None:
    assert generate_samples(count=50, seed=7) == generate_samples(count=50, seed=7)


def test_different_seeds_produce_different_data() -> None:
    assert generate_samples(count=50, seed=7) != generate_samples(count=50, seed=8)


def test_the_requested_number_of_samples_is_produced() -> None:
    assert len(generate_samples(count=137, seed=DEFAULT_SEED)) == 137


def test_an_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_samples(count=0)


def test_every_sample_carries_exactly_the_canonical_features() -> None:
    for sample in generate_samples(count=200, seed=DEFAULT_SEED):
        assert set(sample) == set(FEATURE_ORDER)


def test_every_value_is_a_finite_number() -> None:
    for sample in generate_samples(count=500, seed=DEFAULT_SEED):
        for name, value in sample.items():
            assert isinstance(value, float), name
            assert math.isfinite(value), name


@pytest.mark.parametrize(
    ("feature", "low", "high"),
    [
        pytest.param("soc", 5.0, 100.0, id="soc-percent"),
        pytest.param("battery_voltage", 320.0, 425.0, id="pack-voltage"),
        pytest.param("battery_temperature", -5.0, 80.0, id="pack-temperature"),
        pytest.param("speed", 0.0, 135.0, id="road-speed"),
        pytest.param("motor_rpm", 0.0, 11_500.0, id="motor-speed"),
    ],
)
def test_values_stay_within_the_demo_ranges(feature: str, low: float, high: float) -> None:
    values = list(iter_feature_values(generate_samples(count=2_000, seed=DEFAULT_SEED), feature))

    assert min(values) >= low, feature
    assert max(values) <= high, feature


def test_speed_is_never_negative() -> None:
    """Speed is a magnitude, matching the telemetry contract."""
    values = list(iter_feature_values(generate_samples(count=2_000, seed=DEFAULT_SEED), "speed"))

    assert min(values) >= 0.0


def test_motor_rpm_is_never_negative() -> None:
    values = list(iter_feature_values(generate_samples(count=2_000, seed=DEFAULT_SEED), "motor_rpm"))

    assert min(values) >= 0.0


def test_current_is_negative_at_driving_speed() -> None:
    """Discharging under load.

    Only checked above crawling speed: near 0 km/h the noise term can outweigh
    the small drive current and produce a briefly positive value, which is what
    regenerative braking looks like. Sustained charging happens while stopped.
    """
    driving = [
        sample
        for sample in generate_samples(count=2_000, seed=DEFAULT_SEED)
        if sample["speed"] > 20.0
    ]

    assert driving
    assert all(sample["battery_current"] < 0 for sample in driving)


def test_charging_only_happens_while_stopped() -> None:
    samples = generate_samples(count=2_000, seed=DEFAULT_SEED)
    charging = [sample for sample in samples if sample["battery_current"] > 20.0]

    assert charging
    assert all(sample["speed"] < 1.0 for sample in charging)


def test_motor_speed_tracks_road_speed() -> None:
    """The coupling the model is meant to learn."""
    for sample in generate_samples(count=500, seed=DEFAULT_SEED):
        expected = sample["speed"] * DRIVE_RATIO_RPM_PER_KPH
        assert abs(sample["motor_rpm"] - expected) < 250.0


def test_voltage_rises_with_state_of_charge() -> None:
    samples = generate_samples(count=2_000, seed=DEFAULT_SEED)
    low = [s["battery_voltage"] for s in samples if s["soc"] < 20.0]
    high = [s["battery_voltage"] for s in samples if s["soc"] > 90.0]

    assert sum(low) / len(low) < sum(high) / len(high)


def test_the_default_dataset_is_large_enough_to_train_on() -> None:
    assert DEFAULT_SAMPLE_COUNT >= 1_000


# --------------------------------------------------------------------------- #
# Laying samples out as a matrix
# --------------------------------------------------------------------------- #


def test_the_matrix_follows_the_given_order_not_the_dict_order() -> None:
    sample = {name: float(index) for index, name in enumerate(reversed(FEATURE_ORDER))}

    assert to_matrix([sample], FEATURE_ORDER) == [[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]


def test_the_matrix_has_one_column_per_feature() -> None:
    matrix = to_matrix(generate_samples(count=10, seed=DEFAULT_SEED), FEATURE_ORDER)

    assert len(matrix) == 10
    assert all(len(row) == len(FEATURE_ORDER) for row in matrix)


def test_a_missing_feature_is_an_error_not_a_silent_gap() -> None:
    with pytest.raises(KeyError):
        to_matrix([{"soc": 50.0}], FEATURE_ORDER)
