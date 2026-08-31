"""Service-level errors, free of any web framework."""


class CapabilityNotImplementedError(RuntimeError):
    """A capability defined by the architecture is not implemented yet.

    Raised instead of returning a fabricated prediction or fabricated model
    metadata. The API layer maps it to ``501 Not Implemented``.
    """

    def __init__(self, capability: str, arrives_in: str) -> None:
        super().__init__(f"{capability} is not implemented yet ({arrives_in})")
        self.capability = capability
        self.arrives_in = arrives_in
