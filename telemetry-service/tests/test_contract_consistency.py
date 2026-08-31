"""The API schema and the domain unit tables must not drift apart."""

from typing import get_args

from app.api import schemas
from app.domain.units import CANONICAL_UNITS, SUPPORTED_SOURCE_UNITS, MetricName

MEASUREMENT_MODELS = {
    MetricName.SOC: schemas.SocMeasurement,
    MetricName.BATTERY_VOLTAGE: schemas.BatteryVoltageMeasurement,
    MetricName.BATTERY_CURRENT: schemas.BatteryCurrentMeasurement,
    MetricName.BATTERY_TEMPERATURE: schemas.BatteryTemperatureMeasurement,
    MetricName.SPEED: schemas.SpeedMeasurement,
    MetricName.MOTOR_RPM: schemas.MotorRpmMeasurement,
}


def test_api_accepts_exactly_the_documented_source_units() -> None:
    for metric, model in MEASUREMENT_MODELS.items():
        accepted = get_args(model.model_fields["unit"].annotation)
        assert accepted == SUPPORTED_SOURCE_UNITS[metric], metric


def test_api_metric_set_matches_the_domain_metric_set() -> None:
    assert set(schemas.TelemetryMetrics.model_fields) == {metric.value for metric in MetricName}


def test_every_metric_has_a_canonical_unit_that_is_also_an_accepted_source_unit() -> None:
    for metric in MetricName:
        assert CANONICAL_UNITS[metric] in SUPPORTED_SOURCE_UNITS[metric]


def test_response_metrics_are_canonical_scalars_without_units() -> None:
    """Canonical metrics carry no per-value unit: the schema fixes the unit."""
    assert set(schemas.CanonicalMetrics.model_fields) == {metric.value for metric in MetricName}
    for field in schemas.CanonicalMetrics.model_fields.values():
        assert field.annotation is float
