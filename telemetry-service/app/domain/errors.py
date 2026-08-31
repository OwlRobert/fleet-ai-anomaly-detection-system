"""Domain errors raised when a source event cannot be normalized.

These carry the facts of the rejection, not an HTTP status or an API error
code: mapping them onto the contract's error codes is the API layer's job.
"""

from datetime import datetime, timedelta


class NormalizationError(ValueError):
    """A validated source event cannot be turned into a canonical event."""


class ClockSkewError(NormalizationError):
    """``event_time`` sits outside the accepted window around ``received_at``.

    The event is rejected, never corrected: rewriting a device's timestamp to
    server time destroys the evidence that its clock is wrong and fabricates a
    measurement time.
    """

    def __init__(
        self, message: str, *, event_time: datetime, received_at: datetime, limit: timedelta
    ) -> None:
        super().__init__(message)
        self.event_time = event_time
        self.received_at = received_at
        self.limit = limit


class ClockSkewFutureError(ClockSkewError):
    """``event_time`` is too far ahead of ``received_at``."""


class EventTooOldError(ClockSkewError):
    """``event_time`` is too far behind ``received_at``."""
