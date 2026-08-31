"""Metric identity and unit vocabulary for source telemetry.

This module is the single source of truth for *which* metrics exist and *which*
source units are accepted for each of them. The API schemas are validated
against these tables (see ``tests/test_contract_consistency.py``).

This module states *which* unit is canonical for each metric; the arithmetic
that converts into it lives in ``app.domain.conversions``.
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

SCHEMA_VERSION = "1.0"
"""The only telemetry payload schema version this service accepts."""


class MetricName(StrEnum):
    """The six telemetry metrics of the MVP contract."""

    SOC = "soc"
    BATTERY_VOLTAGE = "battery_voltage"
    BATTERY_CURRENT = "battery_current"
    BATTERY_TEMPERATURE = "battery_temperature"
    SPEED = "speed"
    MOTOR_RPM = "motor_rpm"


CANONICAL_UNITS: Mapping[MetricName, str] = MappingProxyType(
    {
        MetricName.SOC: "percent",
        MetricName.BATTERY_VOLTAGE: "V",
        MetricName.BATTERY_CURRENT: "A",
        MetricName.BATTERY_TEMPERATURE: "degC",
        MetricName.SPEED: "km/h",
        MetricName.MOTOR_RPM: "rpm",
    }
)
"""Internal canonical unit per metric, produced by ``TelemetryNormalizer``."""


SUPPORTED_SOURCE_UNITS: Mapping[MetricName, tuple[str, ...]] = MappingProxyType(
    {
        MetricName.SOC: ("percent", "fraction"),
        MetricName.BATTERY_VOLTAGE: ("V", "mV"),
        MetricName.BATTERY_CURRENT: ("A", "mA"),
        MetricName.BATTERY_TEMPERATURE: ("degC", "degF", "K"),
        MetricName.SPEED: ("km/h", "mph", "m/s"),
        MetricName.MOTOR_RPM: ("rpm",),
    }
)
"""Accepted source units per metric. Matching is case-sensitive by contract."""
