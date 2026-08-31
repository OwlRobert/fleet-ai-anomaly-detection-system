"""The canonical vocabulary this service publishes and validates against."""

from app.api import schemas
from app.domain.features import CANONICAL_UNITS, FEATURE_ORDER

DOCUMENTED_CANONICAL_UNITS = {
    "soc": "percent",
    "battery_voltage": "V",
    "battery_current": "A",
    "battery_temperature": "degC",
    "speed": "km/h",
    "motor_rpm": "rpm",
}


def test_canonical_units_match_the_approved_contract() -> None:
    assert dict(CANONICAL_UNITS) == DOCUMENTED_CANONICAL_UNITS


def test_feature_order_covers_exactly_the_canonical_features() -> None:
    assert set(FEATURE_ORDER) == set(CANONICAL_UNITS)
    assert len(FEATURE_ORDER) == len(CANONICAL_UNITS)


def test_request_schema_covers_exactly_the_canonical_features() -> None:
    assert set(schemas.CanonicalFeatures.model_fields) == set(FEATURE_ORDER)


def test_features_carry_no_unit_of_their_own() -> None:
    """Units are a property of the schema, never of an individual value."""
    for field in schemas.CanonicalFeatures.model_fields.values():
        assert field.annotation is float
