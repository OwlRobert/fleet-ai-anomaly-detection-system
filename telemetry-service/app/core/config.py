"""Telemetry Service configuration.

Only what is actually used.

The two clock-skew bounds are read under the exact names `.env.example`
documents, which carry no service prefix; `validation_alias` bypasses
``env_prefix`` so a documented variable is never silently renamed.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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

    mongodb_uri: SecretStr = Field(
        default=SecretStr("mongodb://localhost:27017"),
        validation_alias="MONGODB_URI",
        min_length=1,
        description="Connection string for the telemetry event store. May carry credentials.",
    )
    mongodb_database: str = Field(
        default="fleet_telemetry",
        validation_alias="MONGODB_DATABASE",
        min_length=1,
        description="Database holding the telemetry collection.",
    )
    mongodb_telemetry_collection: str = Field(
        default="telemetry_events",
        validation_alias="MONGODB_TELEMETRY_COLLECTION",
        min_length=1,
        description="Collection of telemetry event documents.",
    )
    mongodb_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias="MONGODB_TIMEOUT_SECONDS",
        description="Server selection and socket timeout; ingestion fails closed when exceeded.",
    )

    inference_service_url: str = Field(
        default="http://localhost:8001",
        validation_alias="INFERENCE_SERVICE_URL",
        pattern=r"^https?://.+",
        description="Base URL of the Inference Service.",
    )
    inference_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        validation_alias="INFERENCE_TIMEOUT_SECONDS",
        description="Bounded timeout for one inference attempt. No retries.",
    )

    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        pattern=r"^(?i:CRITICAL|ERROR|WARNING|INFO|DEBUG)$",
        description="Root log level.",
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        validation_alias="LOG_FORMAT",
        description="`json` for collectors, `text` for reading locally.",
    )

    @property
    def mongodb_timeout_ms(self) -> int:
        """The driver expects milliseconds."""
        return int(self.mongodb_timeout_seconds * 1000)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return Settings()
