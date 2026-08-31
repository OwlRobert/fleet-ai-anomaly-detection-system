"""Source-unit to canonical-unit conversion.

One small named function per real conversion, dispatched through a per-metric
table. The table is the readable statement of the contract: every accepted
source unit appears exactly once, next to the arithmetic that canonicalizes it.

Conversion happens here and nowhere else — never in a Pydantic validator, never
in a route handler. Phase 1 validation answers "is `mph` an accepted unit for
speed?"; this module answers "what is 32.3 mph in km/h?".

Values are converted in ordinary IEEE-754 double precision and returned
unrounded. Rounding is a display concern and belongs to clients.
"""

from types import MappingProxyType
from typing import Callable, Mapping

from app.domain.units import CANONICAL_UNITS, MetricName


def _identity(value: float) -> float:
    """The source unit is already the canonical unit."""
    return value


def _fraction_to_percent(value: float) -> float:
    return value * 100.0


def _millivolts_to_volts(value: float) -> float:
    return value / 1000.0


def _milliamperes_to_amperes(value: float) -> float:
    return value / 1000.0


def _fahrenheit_to_celsius(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def _kelvin_to_celsius(value: float) -> float:
    return value - 273.15


def _miles_per_hour_to_kph(value: float) -> float:
    return value * 1.609344


def _metres_per_second_to_kph(value: float) -> float:
    return value * 3.6


Converter = Callable[[float], float]

_CONVERTERS: Mapping[MetricName, Mapping[str, Converter]] = MappingProxyType(
    {
        MetricName.SOC: MappingProxyType(
            {"percent": _identity, "fraction": _fraction_to_percent}
        ),
        MetricName.BATTERY_VOLTAGE: MappingProxyType(
            {"V": _identity, "mV": _millivolts_to_volts}
        ),
        MetricName.BATTERY_CURRENT: MappingProxyType(
            {"A": _identity, "mA": _milliamperes_to_amperes}
        ),
        MetricName.BATTERY_TEMPERATURE: MappingProxyType(
            {"degC": _identity, "degF": _fahrenheit_to_celsius, "K": _kelvin_to_celsius}
        ),
        MetricName.SPEED: MappingProxyType(
            {
                "km/h": _identity,
                "mph": _miles_per_hour_to_kph,
                "m/s": _metres_per_second_to_kph,
            }
        ),
        MetricName.MOTOR_RPM: MappingProxyType({"rpm": _identity}),
    }
)


def to_canonical(metric: MetricName, value: float, unit: str) -> float:
    """Convert one measurement into its metric's canonical unit.

    Args:
        metric: Which metric is being converted.
        value: The measured value, expressed in ``unit``.
        unit: The source unit, matched case-sensitively.

    Returns:
        The value in ``CANONICAL_UNITS[metric]``, unrounded and always a
        ``float`` so an integer input cannot leak an ``int`` into the canonical
        model.

    Raises:
        ValueError: If ``unit`` is not an accepted source unit for ``metric``.
            The transport adapter rejects unsupported units long before this
            point, so reaching it means a caller skipped contract validation.
    """
    converters = _CONVERTERS[metric]
    try:
        convert = converters[unit]
    except KeyError:
        raise ValueError(
            f"{unit!r} is not an accepted source unit for {metric.value}; "
            f"accepted: {', '.join(converters)}"
        ) from None
    return float(convert(value))


def canonical_unit(metric: MetricName) -> str:
    """The unit a converted value of this metric is expressed in."""
    return CANONICAL_UNITS[metric]
