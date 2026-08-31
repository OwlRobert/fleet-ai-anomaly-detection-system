"""Inference Service configuration.

Only what Phase 1 actually uses. The model artifact path, model name and model
version documented in `.env.example` belong to model loading, which does not
exist yet and is therefore not wired in here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_NAME = "inference-service"
"""Fixed service identity. Not configuration: it never varies by environment."""


class Settings(BaseSettings):
    """Environment-provided settings, read from ``INFERENCE_SERVICE_*``."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return Settings()
