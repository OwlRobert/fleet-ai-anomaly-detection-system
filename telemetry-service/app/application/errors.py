"""Application-layer errors."""

from app.domain.inference import InferenceErrorCode

class CapabilityNotImplementedError(RuntimeError):
    """A capability defined by the architecture is not implemented yet.

    Raised instead of returning fabricated data, so that an unimplemented
    capability can never be mistaken for a successful result. The API layer maps
    it to ``501 Not Implemented``. It disappears as each capability lands.
    """

    def __init__(self, capability: str, arrives_in: str) -> None:
        super().__init__(f"{capability} is not implemented yet ({arrives_in})")
        self.capability = capability
        self.arrives_in = arrives_in


class PersistenceUnavailableError(RuntimeError):
    """The event store could not be reached, or the write could not complete.

    Ingestion is **fail-closed** on this: the client is told the event was not
    accepted, so its retry — safe by ``event_id`` idempotency — is the recovery
    mechanism. Acknowledging an event that was never stored would lose it.
    """


class InferenceFailedError(RuntimeError):
    """Inference did not return a usable verdict.

    Ingestion is **fail-open** on this: the telemetry is still stored, with
    ``inference.status = FAILED`` and this error code. The message is for the
    logs; only ``error_code`` is ever exposed.
    """

    def __init__(self, error_code: InferenceErrorCode, detail: str) -> None:
        super().__init__(f"{error_code.value}: {detail}")
        self.error_code = error_code


class DuplicateEventIdError(RuntimeError):
    """An event with this ``event_id`` already exists.

    Raised by the repository when the unique constraint rejects a write. It says
    nothing about whether the existing event is the *same* event; the use case
    decides that by comparing logical identity.
    """

    def __init__(self, event_id: str) -> None:
        super().__init__(f"event_id {event_id!r} already exists")
        self.event_id = event_id
