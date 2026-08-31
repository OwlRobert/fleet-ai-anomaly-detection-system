"""The inference verdict attached to a stored telemetry event.

Three states, and the distinction between them is the point:

* ``PENDING``   — the event is stored but has not been scored. Not an anomaly
                  verdict of any kind, and never reported as one.
* ``COMPLETED`` — the model ran and returned a verdict.
* ``FAILED``    — the model was called and did not answer.

An unscored event is **not** a non-anomalous event, so ``is_anomaly`` and
``anomaly_score`` stay ``None`` unless the status is ``COMPLETED``.
"""

from dataclasses import dataclass
from enum import StrEnum


class InferenceStatus(StrEnum):
    """Whether this event has been scored, and how that turned out."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    """What the model said about one event, and which model said it."""

    status: InferenceStatus
    is_anomaly: bool | None = None
    anomaly_score: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    error_code: str | None = None

    @classmethod
    def pending(cls) -> "InferenceOutcome":
        """The state of an event that has been persisted but never scored."""
        return cls(status=InferenceStatus.PENDING)

    @property
    def is_confirmed_anomaly(self) -> bool:
        """True only when a completed run actually said 'anomaly'.

        A pending or failed event is never an anomaly: the absence of a verdict
        is not a negative verdict.
        """
        return self.status is InferenceStatus.COMPLETED and self.is_anomaly is True
