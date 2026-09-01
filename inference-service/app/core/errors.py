"""Service-level errors, free of any web framework."""


class ArtifactLoadError(RuntimeError):
    """The model artifact could not be loaded or is not usable.

    Raised at startup only. The service stays alive without a model — it just
    cannot serve predictions — so this never becomes a fabricated result.

    The message is for the logs. It may name a file path, so it must not be
    echoed to a client.
    """


class ModelNotLoadedError(RuntimeError):
    """A prediction or model description was requested with no model loaded.

    Raised instead of inventing a verdict. The API layer maps it to
    ``503 MODEL_NOT_LOADED``.
    """
