"""Application-layer errors."""


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
