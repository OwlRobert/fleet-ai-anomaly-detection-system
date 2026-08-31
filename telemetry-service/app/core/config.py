"""Telemetry Service configuration.

Only what is actually used. The settings documented in `.env.example` for later
phases (MongoDB, inference client, query limits) are deliberately not wired in
here.

The two clock-skew bounds are read under the exact names `.env.example`
documents, which carry no service prefix; `validation_alias` bypasses
``env_prefix`` so a documented variable is never silently renamed.
"""

from functools import lru_cache

from pydantic import Field
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

    max_clock_skew_future_seconds: int = Field(
        default=300,
        ge=0,
        validation_alias="MAX_CLOCK_SKEW_FUTURE_SECONDS",
        description="How far ahead of received_at an event_time may sit before CLOCK_SKEW_FUTURE.",
    )
    max_event_age_days: int = Field(
        default=30,
        ge=0,
        validation_alias="MAX_EVENT_AGE_DAYS",
        description="How far behind received_at an event_time may sit before EVENT_TOO_OLD.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return Settings()
