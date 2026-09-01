"""The on-disk model artifact: how it is written, and how it is read back.

Writer and reader live in the same module on purpose. The artifact is a plain
``dict`` rather than a pickled custom class, so loading never depends on a class
still existing at the same import path — a version of it moving or being renamed
would otherwise break every previously trained artifact.

Loading validates rather than trusts. An artifact that does not declare exactly
the canonical feature order this service serves is refused: silently scoring a
vector whose columns mean something else is worse than serving nothing.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import sklearn

from app.core.errors import ArtifactLoadError
from app.domain.features import CANONICAL_UNITS, FEATURE_ORDER
from app.domain.prediction import ModelMetadata

ARTIFACT_SCHEMA_VERSION = 1
"""Format of the artifact dict itself, independent of the model version."""

REQUIRED_KEYS = frozenset(
    {
        "artifact_schema_version",
        "model",
        "model_name",
        "model_version",
        "algorithm",
        "feature_order",
        "canonical_units",
        "trained_at",
        "sklearn_version",
        "training",
    }
)


def build_artifact(
    *,
    model: Any,
    model_name: str,
    model_version: str,
    trained_at: datetime,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the dict that gets serialized to disk.

    The feature order and canonical units are taken from this service's own
    vocabulary, so a trained artifact always records the dialect it was built
    against rather than a copy that can drift.
    """
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": model,
        "model_name": model_name,
        "model_version": model_version,
        "algorithm": f"{type(model).__module__.rsplit('._', 1)[0]}.{type(model).__name__}",
        "feature_order": list(FEATURE_ORDER),
        "canonical_units": dict(CANONICAL_UNITS),
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "training": dict(training),
    }


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A validated artifact: the estimator, and what describes it."""

    model: Any
    metadata: ModelMetadata
    training: Mapping[str, Any]


def _digest(path: Path) -> str:
    """SHA-256 of the artifact file, so a deployment can be identified."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact(
    path: Path, *, expected_name: str | None = None, expected_version: str | None = None
) -> LoadedArtifact:
    """Read and validate the artifact at ``path``.

    Args:
        path: Where the joblib file lives.
        expected_name: If given, the artifact must declare this model name.
        expected_version: If given, the artifact must declare this version.
            Together these catch a deployment pointed at the wrong artifact.

    Returns:
        The validated artifact.

    Raises:
        ArtifactLoadError: The file is missing, unreadable, not an artifact of a
            supported schema, or declares a feature vocabulary this service does
            not serve.
    """
    if not path.is_file():
        raise ArtifactLoadError(f"no model artifact at {path}")

    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is the same to us
        raise ArtifactLoadError(f"model artifact at {path} could not be read") from exc

    if not isinstance(payload, dict):
        raise ArtifactLoadError("model artifact is not an artifact dict")

    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ArtifactLoadError(f"model artifact is missing keys: {', '.join(sorted(missing))}")

    schema = payload["artifact_schema_version"]
    if schema != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactLoadError(
            f"model artifact schema {schema!r} is not supported "
            f"(this service reads schema {ARTIFACT_SCHEMA_VERSION})"
        )

    feature_order = tuple(payload["feature_order"])
    if feature_order != FEATURE_ORDER:
        raise ArtifactLoadError(
            f"model artifact feature order {feature_order} does not match the "
            f"canonical order {FEATURE_ORDER}"
        )

    if dict(payload["canonical_units"]) != dict(CANONICAL_UNITS):
        raise ArtifactLoadError("model artifact canonical units do not match this service")

    model = payload["model"]
    if not (hasattr(model, "predict") and hasattr(model, "decision_function")):
        raise ArtifactLoadError("model artifact does not hold a usable estimator")

    if expected_name is not None and payload["model_name"] != expected_name:
        raise ArtifactLoadError(
            f"model artifact is {payload['model_name']!r}, but {expected_name!r} was configured"
        )
    if expected_version is not None and payload["model_version"] != expected_version:
        raise ArtifactLoadError(
            f"model artifact is version {payload['model_version']!r}, "
            f"but {expected_version!r} was configured"
        )

    return LoadedArtifact(
        model=model,
        metadata=ModelMetadata(
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            algorithm=payload["algorithm"],
            feature_order=feature_order,
            canonical_units=dict(payload["canonical_units"]),
            trained_at=payload["trained_at"],
            sklearn_version=payload["sklearn_version"],
            artifact_sha256=_digest(path),
        ),
        training=dict(payload["training"]),
    )
