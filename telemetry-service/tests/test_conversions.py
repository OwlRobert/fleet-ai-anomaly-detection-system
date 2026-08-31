"""Source-unit to canonical-unit conversion.

Every source unit the Phase 1 contract accepts is covered, including the
pass-through cases, so no accepted unit can be silently unimplemented.
"""

import pytest

from app.domain.conversions import to_canonical
from app.domain.units import CANONICAL_UNITS, SUPPORTED_SOURCE_UNITS, MetricName


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(51.9818, "km/h", 51.9818, id="km/h-pass-through"),
        pytest.param(32.3, "mph", 51.9818112, id="mph"),
        pytest.param(0.0, "mph", 0.0, id="mph-zero"),
        pytest.param(14.4394, "m/s", 51.98184, id="m/s"),
    ],
)
def test_speed_converts_to_kph(value: float, unit: str, expected: float) -> None:
    assert to_canonical(MetricName.SPEED, value, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(35.0, "degC", 35.0, id="degC-pass-through"),
        pytest.param(96.4, "degF", 35.77777777777778, id="degF"),
        pytest.param(32.0, "degF", 0.0, id="degF-freezing"),
        pytest.param(-40.0, "degF", -40.0, id="degF-crossover"),
        pytest.param(273.15, "K", 0.0, id="K-freezing"),
        pytest.param(308.15, "K", 35.0, id="K"),
    ],
)
def test_temperature_converts_to_celsius(value: float, unit: str, expected: float) -> None:
    assert to_canonical(MetricName.BATTERY_TEMPERATURE, value, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(72.5, "percent", 72.5, id="percent-pass-through"),
        pytest.param(0.725, "fraction", 72.5, id="fraction"),
        pytest.param(0.0, "fraction", 0.0, id="fraction-empty"),
        pytest.param(1.0, "fraction", 100.0, id="fraction-full"),
    ],
)
def test_soc_converts_to_percent(value: float, unit: str, expected: float) -> None:
    assert to_canonical(MetricName.SOC, value, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(396.2, "V", 396.2, id="V-pass-through"),
        pytest.param(396200.0, "mV", 396.2, id="mV"),
    ],
)
def test_voltage_converts_to_volts(value: float, unit: str, expected: float) -> None:
    assert to_canonical(MetricName.BATTERY_VOLTAGE, value, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(-14.7, "A", -14.7, id="A-pass-through"),
        pytest.param(-14700.0, "mA", -14.7, id="mA"),
    ],
)
def test_current_converts_to_amperes(value: float, unit: str, expected: float) -> None:
    assert to_canonical(MetricName.BATTERY_CURRENT, value, unit) == pytest.approx(expected)


def test_motor_rpm_is_already_canonical() -> None:
    assert to_canonical(MetricName.MOTOR_RPM, 4120.0, "rpm") == pytest.approx(4120.0)


def test_negative_motor_rpm_passes_through_unchanged() -> None:
    """No rotation-direction convention is documented, so none is imposed."""
    assert to_canonical(MetricName.MOTOR_RPM, -50.0, "rpm") == pytest.approx(-50.0)


def test_every_accepted_source_unit_has_a_conversion() -> None:
    """A unit the contract accepts but cannot convert would be a silent gap."""
    for metric, units in SUPPORTED_SOURCE_UNITS.items():
        for unit in units:
            assert isinstance(to_canonical(metric, 1.0, unit), float), (metric, unit)


def test_canonical_units_are_pass_through_for_every_metric() -> None:
    for metric, unit in CANONICAL_UNITS.items():
        assert to_canonical(metric, 12.25, unit) == pytest.approx(12.25)


def test_conversion_result_is_always_a_float() -> None:
    """An int input must not leak an int into the canonical model."""
    assert isinstance(to_canonical(MetricName.MOTOR_RPM, 4120, "rpm"), float)


def test_unsupported_unit_is_rejected() -> None:
    """Reaching this means a caller skipped contract validation."""
    with pytest.raises(ValueError, match="not an accepted source unit"):
        to_canonical(MetricName.SPEED, 10.0, "knots")


def test_values_are_not_rounded() -> None:
    """Rounding is a display concern; the domain keeps full precision."""
    assert to_canonical(MetricName.SPEED, 32.3, "mph") == 32.3 * 1.609344
