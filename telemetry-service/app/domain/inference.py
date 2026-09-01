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


class InferenceErrorCode(StrEnum):
    """Why inference did not produce a verdict.

    One code per failure class, stable enough to alert on. They describe *our*
    view of the failure, never the upstream's internals: no exception class
    names, hostnames, URLs or response bodies.
    """

    TIMEOUT = "INFERENCE_TIMEOUT"
    UNAVAILABLE = "INFERENCE_UNAVAILABLE"
    UNREACHABLE = "INFERENCE_UNREACHABLE"
    INVALID_RESPONSE = "INFERENCE_INVALID_RESPONSE"


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
        """An event persisted before anything could score it.

        Written by the phase that had persistence but no inference. The
        synchronous write path no longer produces it: every event now finishes
        as ``COMPLETED`` or ``FAILED``. It is kept so records written earlier
        still read back, and so an asynchronous write path could use it again.
        """
        return cls(status=InferenceStatus.PENDING)

    @classmethod
    def completed(
        cls,
        *,
        is_anomaly: bool,
        anomaly_score: float,
        model_name: str,
        model_version: str,
    ) -> "InferenceOutcome":
        """A verdict the model actually returned."""
        return cls(
            status=InferenceStatus.COMPLETED,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            model_name=model_name,
            model_version=model_version,
        )

    @classmethod
    def failed(cls, error_code: InferenceErrorCode) -> "InferenceOutcome":
        """Inference did not complete.

        Every verdict field stays ``None``. Defaulting ``is_anomaly`` to false
        here would record a false negative as if the model had spoken, and
        naming a model that never answered would be worse still.
        """
        return cls(status=InferenceStatus.FAILED, error_code=error_code.value)

    @property
    def is_confirmed_anomaly(self) -> bool:
        """True only when a completed run actually said 'anomaly'.

        A pending or failed event is never an anomaly: the absence of a verdict
        is not a negative verdict.
        """
        return self.status is InferenceStatus.COMPLETED and self.is_anomaly is True
