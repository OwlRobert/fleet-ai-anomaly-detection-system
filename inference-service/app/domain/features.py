"""The canonical feature vocabulary this service scores.

The Inference Service is unit-agnostic by design: it accepts canonical values
only and performs no conversion. These tables are the contract that lets a
caller check it is speaking the same dialect, and they are what `/model/info`
publishes.

The Telemetry Service holds an equivalent canonical table. The duplication is
deliberate: the two services are independently deployable, and this is a shared
*contract*, not shared code.
"""

from types import MappingProxyType
from typing import Mapping

FEATURE_ORDER: tuple[str, ...] = (
    "soc",
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "speed",
    "motor_rpm",
)
"""Fixed feature order. A reordering is a model version change, not a refactor."""


CANONICAL_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "soc": "percent",
        "battery_voltage": "V",
        "battery_current": "A",
        "battery_temperature": "degC",
        "speed": "km/h",
        "motor_rpm": "rpm",
    }
)
"""The unit each feature is expected in. Source units never reach this service."""
