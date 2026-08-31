"""Pydantic contracts for the Inference Service.

The defining property of these schemas is what they **refuse**. A feature is a
bare canonical number: `{"speed": 51.98}`. A source measurement object such as
`{"speed": {"value": 32.3, "unit": "mph"}}` is rejected, and so is any `unit` or
`source_units` key. Unit conversion happens once, in the Telemetry Service,
before a request ever reaches here — so the model can never depend on a
client's display preference.

Strings are rejected where numbers are expected, which keeps the API
locale-independent: `"1,5"` is not a number.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.domain.features import CANONICAL_UNITS

CanonicalValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]
"""A finite JSON number. Strings, booleans, NaN and Infinity are rejected."""


class CanonicalFeatures(BaseModel):
    """One feature vector, already in canonical units.

    Units are a property of this schema and are never carried per value:
    `soc` percent, `battery_voltage` V, `battery_current` A,
    `battery_temperature` degC, `speed` km/h, `motor_rpm` rpm.
    """

    model_config = ConfigDict(extra="forbid")

    soc: CanonicalValue = Field(description=f"State of charge in {CANONICAL_UNITS['soc']}.")
    battery_voltage: CanonicalValue = Field(description=f"Pack voltage in {CANONICAL_UNITS['battery_voltage']}.")
    battery_current: CanonicalValue = Field(description=f"Pack current in {CANONICAL_UNITS['battery_current']}.")
    battery_temperature: CanonicalValue = Field(
        description=f"Pack temperature in {CANONICAL_UNITS['battery_temperature']}."
    )
    speed: CanonicalValue = Field(description=f"Vehicle speed in {CANONICAL_UNITS['speed']}.")
    motor_rpm: CanonicalValue = Field(description=f"Motor speed in {CANONICAL_UNITS['motor_rpm']}.")


class PredictionRequest(BaseModel):
    """Score exactly one canonical feature vector."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "features": {
                    "soc": 78.5,
                    "battery_voltage": 396.2,
                    "battery_current": -14.7,
                    "battery_temperature": 35.7778,
                    "speed": 51.9818,
                    "motor_rpm": 4120.0,
                }
            }
        },
    )

    features: CanonicalFeatures


class PredictionResponse(BaseModel):
    """The model's verdict, and which model produced it."""

    model_config = ConfigDict(protected_namespaces=())

    is_anomaly: bool = Field(description="True when the model classifies this vector as anomalous.")
    anomaly_score: float = Field(description="Anomaly score from the model. Lower values are more anomalous.")
    model_name: str = Field(description="Name of the model that produced this result.")
    model_version: str = Field(description="Version of the model that produced this result.")


class ModelInfoResponse(BaseModel):
    """Metadata of the loaded model artifact.

    `feature_order` and `canonical_units` are contract, not documentation: they
    let a caller verify it is speaking the same dialect as the loaded model.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    algorithm: str = Field(description="Fully qualified estimator class.")
    trained_at: datetime = Field(description="When the artifact was trained, in UTC.")
    feature_order: list[str] = Field(description="Feature vector order expected by the model.")
    canonical_units: dict[str, str] = Field(description="Canonical unit per feature.")
    artifact_sha256: str = Field(description="Digest of the loaded artifact file.")
    sklearn_version: str = Field(description="scikit-learn version the artifact was trained with.")
