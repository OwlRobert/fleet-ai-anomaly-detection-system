"""Synthetic training data for the anomaly model.

Real fleet telemetry is not available for this project, so the model is trained
on generated samples of *normal* operation. Everything here is a demonstration
assumption: the ranges and the relationships between features are plausible for
an EV, but they are **not validated against any production EV platform** and
should not be read as vehicle specifications.

The generator is deterministic — the same seed always produces the same rows —
and uses only the standard library, so training needs no network access and no
external dataset.

Some features are deliberately *coupled*, because that is what makes the model
worth training at all:

* pack voltage rises with state of charge, and sags a little under load;
* motor speed tracks road speed through a fixed drive ratio;
* current is negative (discharging) when moving and mildly positive
  (charging) only when stopped;
* pack temperature rises with the magnitude of the current.

A model trained on marginal ranges alone would only ever catch out-of-range
values. Trained on the coupling, it can also flag a sample whose values are each
individually plausible but jointly impossible — 0 km/h at 9000 rpm, say.
"""

import random
from typing import Iterator, Mapping, Sequence

DEFAULT_SAMPLE_COUNT = 5_000
"""Enough variation for IsolationForest without being slow to train."""

DEFAULT_SEED = 20260901
"""Fixed so a rebuild reproduces the same dataset."""

DRIVE_RATIO_RPM_PER_KPH = 82.0
"""Motor revolutions per minute for each km/h of road speed."""

PACK_VOLTAGE_EMPTY = 340.0
PACK_VOLTAGE_FULL = 410.0
"""Pack voltage at 0% and 100% state of charge."""


def _voltage_for(soc: float, current: float, rng: random.Random) -> float:
    """Open-circuit voltage for this charge level, sagging under load."""
    open_circuit = PACK_VOLTAGE_EMPTY + (PACK_VOLTAGE_FULL - PACK_VOLTAGE_EMPTY) * (soc / 100.0)
    sag = 0.0009 * current  # current is negative while discharging, so this drops voltage
    return open_circuit + sag + rng.gauss(0.0, 0.8)


def _current_for(speed: float, rng: random.Random) -> float:
    """Discharge current while moving; occasional charging while stopped."""
    if speed < 1.0:
        if rng.random() < 0.35:
            return rng.uniform(20.0, 120.0)  # charging
        return rng.gauss(-2.0, 1.5)  # parked, small auxiliary draw
    return -(3.0 * speed + rng.gauss(0.0, 12.0))


def _temperature_for(current: float, rng: random.Random) -> float:
    """Pack temperature rises with the magnitude of the current."""
    return 18.0 + 0.055 * abs(current) + rng.gauss(0.0, 2.0)


def generate_samples(
    count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED
) -> list[dict[str, float]]:
    """Generate ``count`` samples of normal-looking operation.

    Args:
        count: How many samples to produce.
        seed: Seed for the pseudo-random generator. The same seed always
            produces the same samples.

    Returns:
        One dict per sample, keyed by canonical feature name, in canonical
        units: percent, V, A, degC, km/h, rpm.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)
    return [_one_sample(rng) for _ in range(count)]


def _one_sample(rng: random.Random) -> dict[str, float]:
    soc = rng.uniform(5.0, 100.0)
    speed = 0.0 if rng.random() < 0.25 else rng.uniform(0.0, 130.0)
    current = _current_for(speed, rng)
    return {
        "soc": soc,
        "battery_voltage": _voltage_for(soc, current, rng),
        "battery_current": current,
        "battery_temperature": _temperature_for(current, rng),
        "speed": speed,
        "motor_rpm": max(0.0, speed * DRIVE_RATIO_RPM_PER_KPH + rng.gauss(0.0, 40.0)),
    }


def to_matrix(
    samples: Sequence[Mapping[str, float]], feature_order: Sequence[str]
) -> list[list[float]]:
    """Lay samples out as rows in ``feature_order``.

    This is the only place a dict becomes a vector. Going through the explicit
    order means the caller's key order can never decide the column order.

    Raises:
        KeyError: If a sample is missing one of the ordered features.
    """
    return [[float(sample[feature]) for feature in feature_order] for sample in samples]


def iter_feature_values(
    samples: Sequence[Mapping[str, float]], feature: str
) -> Iterator[float]:
    """Every value of one feature across the samples."""
    return (sample[feature] for sample in samples)


if __name__ == "__main__":  # pragma: no cover - convenience for eyeballing the data
    for row in generate_samples(count=5):
        print({name: round(value, 3) for name, value in row.items()})
