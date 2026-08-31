"""Pydantic request/response contracts for the Telemetry Service.

Two things this module deliberately does *not* do:

* **No unit conversion.** A measurement is accepted in its source unit and kept
  in it. Converting mph to km/h is `TelemetryNormalizer`'s job, downstream.
* **No time normalization.** ``event_time`` must be timezone-aware, and the
  offset the client sent is preserved. Conversion to UTC and the stamping of
  ``received_at`` happen in the normalizer, not here.

Response models describe the contract these endpoints will fulfil once
persistence and inference exist. They are published in OpenAPI so the contract
is legible now; no handler fabricates one in Phase 1.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

from app.domain.telemetry import Measurement as DomainMeasurement
from app.domain.telemetry import SourceTelemetryEvent
from app.domain.stored import StoredTelemetryEvent
from app.domain.units import SCHEMA_VERSION, MetricName

# --------------------------------------------------------------------------- #
# Shared field types
# --------------------------------------------------------------------------- #


def _reject_blank(value: Any) -> Any:
    """Reject identifiers that are empty or only whitespace."""
    if isinstance(value, str) and not value.strip():
        raise ValueError("identifier must not be blank")
    return value


Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    BeforeValidator(_reject_blank),
]
"""An opaque UTF-8 identifier. The service never parses meaning out of one."""


def _require_string(value: Any) -> Any:
    """Reject non-string timestamps such as epoch numbers."""
    if not isinstance(value, str):
        raise ValueError("event_time must be an ISO-8601 string")
    return value


EventTime = Annotated[AwareDatetime, BeforeValidator(_require_string)]
"""Timezone-aware ISO-8601 timestamp. Naive input is rejected, not assumed."""


MetricValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]
"""A finite JSON number. Strings, booleans, NaN and Infinity are rejected."""


# --------------------------------------------------------------------------- #
# Source measurements - one model per metric, because units are metric-specific
# --------------------------------------------------------------------------- #


class Measurement(BaseModel):
    """A measured value with the explicit unit the device reported it in.

    There is no default unit and no request-level unit system: a device may
    report mph for speed and degC for temperature in the same payload.
    """

    model_config = ConfigDict(extra="forbid")

    value: MetricValue = Field(description="Measured value, expressed in `unit`.")
    unit: str = Field(description="Source unit of this measurement.")


class SocMeasurement(Measurement):
    """State of charge. Range depends on the unit it was reported in."""

    unit: Literal["percent", "fraction"] = Field(description="`percent` (0-100) or `fraction` (0-1).")

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        upper = 100.0 if self.unit == "percent" else 1.0
        if not 0.0 <= self.value <= upper:
            raise PydanticCustomError(
                "soc_out_of_range",
                "soc in {unit} must be between 0 and {upper}",
                {"unit": self.unit, "upper": upper},
            )
        return self


class BatteryVoltageMeasurement(Measurement):
    """Pack voltage."""

    unit: Literal["V", "mV"] = Field(description="Volts or millivolts.")


class BatteryCurrentMeasurement(Measurement):
    """Pack current. Negative values represent discharge."""

    unit: Literal["A", "mA"] = Field(description="Amperes or milliamperes.")


class BatteryTemperatureMeasurement(Measurement):
    """Pack temperature."""

    unit: Literal["degC", "degF", "K"] = Field(description="Celsius, Fahrenheit or Kelvin.")


class SpeedMeasurement(Measurement):
    """Vehicle speed as a magnitude, so it is never negative."""

    value: MetricValue = Field(ge=0, description="Speed magnitude, expressed in `unit`.")
    unit: Literal["km/h", "mph", "m/s"] = Field(description="Kilometres per hour, miles per hour or metres per second.")


class MotorRpmMeasurement(Measurement):
    """Motor rotational speed.

    No sign constraint: the contract does not document a direction convention,
    so one is not invented here.
    """

    unit: Literal["rpm"] = Field(description="Revolutions per minute.")


class TelemetryMetrics(BaseModel):
    """All six metrics of the MVP contract. Unknown metric keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    soc: SocMeasurement
    battery_voltage: BatteryVoltageMeasurement
    battery_current: BatteryCurrentMeasurement
    battery_temperature: BatteryTemperatureMeasurement
    speed: SpeedMeasurement
    motor_rpm: MotorRpmMeasurement


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


class TelemetryIngestRequest(BaseModel):
    """One telemetry event exactly as a device or the simulator emits it."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "schema_version": SCHEMA_VERSION,
                "event_id": "3f0a9c2e-6f4b-4a6f-9d6e-2b1c8f4a77e1",
                "vehicle_id": "veh-tw-0142",
                "site_id": "site-taipei-01",
                "event_time": "2026-08-31T09:14:22.481+08:00",
                "metrics": {
                    "soc": {"value": 78.5, "unit": "percent"},
                    "battery_voltage": {"value": 396.2, "unit": "V"},
                    "battery_current": {"value": -14.7, "unit": "A"},
                    "battery_temperature": {"value": 96.4, "unit": "degF"},
                    "speed": {"value": 32.3, "unit": "mph"},
                    "motor_rpm": {"value": 4120, "unit": "rpm"},
                },
            }
        },
    )

    schema_version: Literal["1.0"] = Field(description="Payload schema version. Only `1.0` is accepted.")
    event_id: Identifier = Field(
        description=(
            "Globally unique domain identifier of this emitted event and the "
            "idempotency key. Opaque to the service; not a storage key."
        )
    )
    vehicle_id: Identifier = Field(description="Opaque vehicle identifier.")
    site_id: Identifier = Field(description="Opaque site identifier.")
    event_time: EventTime = Field(
        description="Device measurement time. Must carry a UTC offset; naive timestamps are rejected."
    )
    metrics: TelemetryMetrics = Field(description="The six metrics, each with an explicit source unit.")

    def to_domain_event(self) -> SourceTelemetryEvent:
        """Translate the transport payload into the domain event.

        A straight mapping: values and units are carried across untouched, and
        the offset on ``event_time`` is preserved. Normalization happens after
        this point, inside the domain.
        """
        return SourceTelemetryEvent(
            schema_version=self.schema_version,
            event_id=self.event_id,
            vehicle_id=self.vehicle_id,
            site_id=self.site_id,
            event_time=self.event_time,
            metrics={
                MetricName(name): DomainMeasurement(value=measurement.value, unit=measurement.unit)
                for name, measurement in self.metrics
            },
        )


# --------------------------------------------------------------------------- #
# Query parameters
# --------------------------------------------------------------------------- #


class TimeRangeQuery(BaseModel):
    """Time range and page size shared by both vehicle history endpoints."""

    model_config = ConfigDict(extra="forbid")

    start: EventTime = Field(description="Inclusive lower bound on `event_time`. Must be timezone-aware.")
    end: EventTime = Field(description="Exclusive upper bound on `event_time`. Must be timezone-aware.")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of events to return.")

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.start >= self.end:
            raise PydanticCustomError("invalid_time_range", "start must be strictly before end")
        return self


# --------------------------------------------------------------------------- #
# Responses - the contract later phases fulfil, published for OpenAPI
# --------------------------------------------------------------------------- #


class CanonicalMetrics(BaseModel):
    """Metrics after normalization, in canonical units.

    Units are a property of this schema, not of each value: `soc` percent,
    `battery_voltage` V, `battery_current` A, `battery_temperature` degC,
    `speed` km/h, `motor_rpm` rpm.
    """

    soc: float
    battery_voltage: float
    battery_current: float
    battery_temperature: float
    speed: float
    motor_rpm: float


class InferenceOutcome(BaseModel):
    """What the model said about this event, and which model said it.

    Three states:

    * ``PENDING``   — the event is stored but has not been scored.
    * ``COMPLETED`` — the model ran and returned a verdict.
    * ``FAILED``    — the model was called and did not answer.

    Only ``COMPLETED`` carries a verdict. Under ``PENDING`` and ``FAILED`` every
    other field is null, because an unscored event is not a non-anomalous event
    and is never reported as one.
    """

    status: Literal["PENDING", "COMPLETED", "FAILED"]
    is_anomaly: bool | None = None
    anomaly_score: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    error_code: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class TelemetryEventResponse(BaseModel):
    """A stored telemetry event, with canonical metrics and its inference result."""

    event_id: str
    vehicle_id: str
    site_id: str
    event_time: datetime = Field(description="Device measurement time, normalized to UTC.")
    received_at: datetime = Field(description="Arrival time stamped by this service, in UTC.")
    duplicate: bool = Field(description="True when this event_id was already stored; the original is returned unchanged.")
    metrics: CanonicalMetrics
    inference: InferenceOutcome

    @classmethod
    def from_stored(cls, stored: StoredTelemetryEvent, *, duplicate: bool) -> "TelemetryEventResponse":
        """Render a stored event for the wire.

        Storage identity never appears here: the response carries ``event_id``
        and nothing MongoDB assigned.
        """
        event = stored.event
        return cls(
            event_id=event.event_id,
            vehicle_id=event.vehicle_id,
            site_id=event.site_id,
            event_time=event.event_time,
            received_at=event.received_at,
            duplicate=duplicate,
            metrics=CanonicalMetrics(
                **{metric.value: value for metric, value in event.metrics.items()}
            ),
            inference=InferenceOutcome(
                status=stored.inference.status.value,
                is_anomaly=stored.inference.is_anomaly,
                anomaly_score=stored.inference.anomaly_score,
                model_name=stored.inference.model_name,
                model_version=stored.inference.model_version,
                error_code=stored.inference.error_code,
            ),
        )


class VehicleTelemetryPage(BaseModel):
    """A page of vehicle events, ordered by `event_time` descending."""

    vehicle_id: str
    start: datetime
    end: datetime
    count: int
    items: list[TelemetryEventResponse]

    @classmethod
    def from_stored(
        cls,
        vehicle_id: str,
        start: datetime,
        end: datetime,
        events: list[StoredTelemetryEvent],
    ) -> "VehicleTelemetryPage":
        """Render a repository result as a page.

        ``duplicate`` is false throughout: these are reads, not ingestions.
        """
        items = [TelemetryEventResponse.from_stored(event, duplicate=False) for event in events]
        return cls(vehicle_id=vehicle_id, start=start, end=end, count=len(items), items=items)
