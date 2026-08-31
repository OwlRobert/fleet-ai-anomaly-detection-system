"""Telemetry Service configuration.

Only what Phase 1 actually uses. The unused settings documented in
`.env.example` for later phases (MongoDB, inference client, clock skew, query
limits) are deliberately not wired in here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_NAME = "telemetry-service"
"""Fixed service identity. Not configuration: it never varies by environment."""


class Settings(BaseSettings):
    """Environment-provided settings, read from ``TELEMETRY_SERVICE_*``."""

    model_config = SettingsConfigDict(
        env_prefix="TELEMETRY_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return Settings()
