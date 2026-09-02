"""Telemetry event generation for the demo simulator.

Everything here is **synthetic demonstration data**, not a validated EV
simulation. Values are plausible enough to exercise the pipeline and nothing
more; no claim is made that they model a real vehicle.

Generation deliberately mirrors the coupling the model was trained on — pack
voltage follows state of charge, motor speed follows road speed, current
follows load — so that "normal" events look normal to the model and an injected
anomaly looks unusual for a reason, rather than by being merely out of range.

Events are emitted in **source units** with an explicit unit per metric, and
`event_time` carries the site's real UTC offset. That is deliberate: it makes
the demo exercise unit normalization and UTC conversion rather than sending
values that are already canonical.
"""

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "1.0"

DRIVE_RATIO_RPM_PER_KPH = 82.0
PACK_VOLTAGE_EMPTY = 340.0
PACK_VOLTAGE_FULL = 410.0

METRIC_NAMES = (
    "soc",
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "speed",
    "motor_rpm",
)
"""The six metrics the Telemetry Service accepts, in the project's fixed order."""


@dataclass(frozen=True, slots=True)
class GeneratedEvent:
    """One event, and whether it was generated as an anomaly candidate.

    ``injected`` records the simulator's *intent*. It is never sent to the
    service and never compared against the verdict: the model decides what is
    anomalous, and the simulator only reports what came back.
    """

    payload: dict[str, Any]
    vehicle_id: str
    event_id: str
    injected: bool


def vehicle_ids(count: int) -> list[str]:
    """``EV-001``, ``EV-002``, … for the requested fleet size."""
    if count < 1:
        raise ValueError("count must be at least 1")
    return [f"EV-{index:03d}" for index in range(1, count + 1)]


def _measurement(value: float, unit: str) -> dict[str, Any]:
    return {"value": round(value, 4), "unit": unit}


def _normal_metrics(rng: random.Random) -> dict[str, dict[str, Any]]:
    """A plausible moment of ordinary driving or parking.

    Sent in mixed source units — mph and degF alongside percent and V — because
    a real fleet mixes conventions, and because it proves normalization runs.
    """
    soc = rng.uniform(15.0, 95.0)
    stopped = rng.random() < 0.25
    speed_kph = 0.0 if stopped else rng.uniform(5.0, 120.0)

    if stopped:
        current = rng.uniform(20.0, 110.0) if rng.random() < 0.35 else rng.gauss(-2.0, 1.5)
    else:
        current = -(3.0 * speed_kph + rng.gauss(0.0, 12.0))

    voltage = (
        PACK_VOLTAGE_EMPTY
        + (PACK_VOLTAGE_FULL - PACK_VOLTAGE_EMPTY) * (soc / 100.0)
        + 0.0009 * current
        + rng.gauss(0.0, 0.8)
    )
    temperature_c = 18.0 + 0.055 * abs(current) + rng.gauss(0.0, 2.0)
    rpm = max(0.0, speed_kph * DRIVE_RATIO_RPM_PER_KPH + rng.gauss(0.0, 40.0))

    return {
        "soc": _measurement(soc, "percent"),
        "battery_voltage": _measurement(voltage, "V"),
        "battery_current": _measurement(current, "A"),
        # Reported in Fahrenheit, so the service has to convert it.
        "battery_temperature": _measurement(temperature_c * 9.0 / 5.0 + 32.0, "degF"),
        # Reported in mph, likewise.
        "speed": _measurement(speed_kph / 1.609344, "mph"),
        "motor_rpm": _measurement(rpm, "rpm"),
    }


def _anomaly_metrics(rng: random.Random) -> dict[str, dict[str, Any]]:
    """A deliberately implausible reading, for demonstration only.

    Two shapes, both far from the training distribution: a pack in obvious
    distress, and a reading that is internally contradictory — stationary while
    the motor spins hard. Neither is a validated real-world fault signature, and
    neither is guaranteed to be flagged: the model decides.
    """
    if rng.random() < 0.5:
        return {
            "soc": _measurement(rng.uniform(70.0, 90.0), "percent"),
            "battery_voltage": _measurement(rng.uniform(240.0, 265.0), "V"),
            "battery_current": _measurement(rng.uniform(-950.0, -820.0), "A"),
            "battery_temperature": _measurement(rng.uniform(180.0, 195.0), "degF"),
            "speed": _measurement(rng.uniform(4.0, 9.0), "mph"),
            "motor_rpm": _measurement(rng.uniform(600.0, 950.0), "rpm"),
        }
    return {
        "soc": _measurement(rng.uniform(60.0, 85.0), "percent"),
        "battery_voltage": _measurement(rng.uniform(385.0, 400.0), "V"),
        "battery_current": _measurement(rng.uniform(-30.0, -5.0), "A"),
        "battery_temperature": _measurement(rng.uniform(68.0, 78.0), "degF"),
        # Stationary at 9000 rpm: each value ordinary, the combination impossible.
        "speed": _measurement(0.0, "mph"),
        "motor_rpm": _measurement(rng.uniform(8500.0, 9500.0), "rpm"),
    }


def generate_event(
    *,
    vehicle_id: str,
    site_id: str,
    rng: random.Random,
    site_timezone: ZoneInfo,
    inject_anomaly: bool,
    now: datetime | None = None,
) -> GeneratedEvent:
    """Build one telemetry event for the given vehicle.

    Args:
        vehicle_id: Which simulated vehicle emitted it.
        site_id: The site the fleet is parked at.
        rng: Seeded generator, so a run is reproducible.
        site_timezone: IANA zone used to stamp ``event_time`` with a real local
            offset, rather than pretending everything happens in UTC.
        inject_anomaly: Generate an anomaly candidate instead of a normal event.
        now: Override the clock, for tests.

    Returns:
        The event and the intent behind it.
    """
    moment = (now or datetime.now(timezone.utc)).astimezone(site_timezone)
    # A second or two in the past: comfortably inside the service's clock-skew
    # window, and closer to how a device would actually report.
    moment -= timedelta(seconds=rng.uniform(0.5, 3.0))

    event_id = str(uuid.uuid4())
    metrics = _anomaly_metrics(rng) if inject_anomaly else _normal_metrics(rng)

    return GeneratedEvent(
        payload={
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "vehicle_id": vehicle_id,
            "site_id": site_id,
            "event_time": moment.isoformat(),
            "metrics": metrics,
        },
        vehicle_id=vehicle_id,
        event_id=event_id,
        injected=inject_anomaly,
    )


def summarize_response(status: int, body: Mapping[str, Any] | None) -> str:
    """One line describing what the service actually said.

    Nothing here is inferred. If the service did not return a verdict, none is
    printed — the simulator never fills in what the model did not say.
    """
    if body is None:
        return f"{status} | no response body"

    if "error" in body:
        return f"{status} | {body['error'].get('code', 'ERROR')}"

    inference = body.get("inference") or {}
    parts = [str(status), str(inference.get("status", "?"))]

    is_anomaly = inference.get("is_anomaly")
    parts.append("anomaly=—" if is_anomaly is None else f"anomaly={str(is_anomaly).lower()}")

    score = inference.get("anomaly_score")
    parts.append("score=—" if score is None else f"score={score:+.4f}")

    if body.get("duplicate"):
        parts.append("duplicate")
    if inference.get("error_code"):
        parts.append(str(inference["error_code"]))
    return " | ".join(parts)
