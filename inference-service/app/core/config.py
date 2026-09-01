"""Inference Service configuration.

Only what is actually used. The artifact path, model name and model version are
read under the exact names `.env.example` documents, which carry no service
prefix; ``validation_alias`` bypasses ``env_prefix`` so a documented variable is
never silently renamed.

The configured name and version are *expectations*, checked against the artifact
at load time so a deployment pointed at the wrong file fails loudly. What
`/model/info` publishes always comes from the artifact itself, never from these.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_NAME = "inference-service"
"""Fixed service identity. Not configuration: it never varies by environment."""


class Settings(BaseSettings):
    """Environment-provided settings, read from ``INFERENCE_SERVICE_*``."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_SERVICE_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    version: str = "0.1.0"

    model_artifact_path: Path = Field(
        default=Path("ml/artifacts/isolation_forest_v0_1_0.joblib"),
        validation_alias="MODEL_ARTIFACT_PATH",
        description="joblib artifact loaded once at startup.",
    )
    model_name: str = Field(
        default="isolation-forest-telemetry",
        validation_alias="MODEL_NAME",
        description="Model name the artifact is required to declare.",
    )
    model_version: str = Field(
        default="0.1.0",
        validation_alias="MODEL_VERSION",
        description="Model version the artifact is required to declare.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return Settings()
