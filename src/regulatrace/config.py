"""Configuration management for Regulatrace using pydantic-settings.

Centralizes environment-driven configuration with safe representation output.
"""

import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

SENSITIVE_KEY_SUBSTRINGS = ("key", "secret", "password", "token", "auth", "credential")


class Settings(BaseSettings):
    """Core application settings for Regulatrace.

    Loads from environment variables prefixed with REGULATRACE_ or an optional .env file.
    """

    model_config = SettingsConfigDict(
        env_prefix="REGULATRACE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Regulatrace"
    # Optional built React assets; unset keeps the existing API-only application.
    frontend_dist: Path | None = None
    environment: Literal["development", "staging", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    debug: bool = False
    # Explicit frontend origins, supplied as a JSON list in the environment.
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    # Required for authentication: 32 random bytes encoded as 64 hex characters.
    jwt_secret: SecretStr | None = None
    # A bounded transition accepts already-issued tokens after an application rename.
    previous_token_issuer: str | None = None
    previous_token_valid_until: int = Field(default=0, ge=0)

    # Optional, explicitly enabled tracing; never include credentials in trace metadata.
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # Model and Provider Settings
    llm_provider: str = "mock"
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_execution: Literal["local", "space"] = "local"
    model_worker_url: str = "https://abdul547-regulatrace.hf.space"
    model_worker_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "REGULATRACE_MODEL_WORKER_SECRET", "REGULATRACE_MODEL_WORKER_KEY", "model_worker_key"
        ),
    )
    hf_token: SecretStr | None = None
    gemini_force_ipv4: bool = False
    # Explicit model-matched USD prices; no assumed provider tariff.
    llm_pricing_model: str | None = None
    llm_input_price_per_million: Decimal | None = Field(default=None, ge=0)
    llm_output_price_per_million: Decimal | None = Field(default=None, ge=0)

    # Sensitive API Credentials (masked in safe_dump/safe_repr)

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("REGULATRACE_GROQ_API_KEY", "groq_api_key"),
    )

    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Database Settings
    database_url: str = "postgresql://localhost:5432/regulatrace"
    database_null_pool: bool = False
    chat_rate_limit_store: Literal["memory", "postgres"] = "memory"

    # Qdrant Vector Store Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "regulatrace_chunks"
    bm25_collection: str = "regulatrace_bm25"
    hybrid_collection: str = "regulatrace_hybrid"
    embedding_batch_size: int = 32

    def safe_dump(self) -> dict[str, Any]:
        """Return a dictionary representation with sensitive values masked."""
        data: dict[str, Any] = {}
        for key, value in self.model_dump().items():
            if isinstance(value, SecretStr) or any(
                sensitive in key.lower() for sensitive in SENSITIVE_KEY_SUBSTRINGS
            ):
                data[key] = "**********"
            elif key == "database_url":
                try:
                    url = make_url(value)
                    secret_query = {
                        name: "**********"
                        for name in url.query
                        if any(sensitive in name.lower() for sensitive in SENSITIVE_KEY_SUBSTRINGS)
                    }
                    if secret_query:
                        url = url.update_query_dict(secret_query)
                    data[key] = url.render_as_string(hide_password=True)
                except (ArgumentError, ValueError, TypeError):
                    data[key] = "**********"
            else:
                data[key] = value
        return data

    def safe_repr(self) -> str:
        """Return a formatted, safe string representation of settings."""
        return json.dumps(self.safe_dump(), indent=2, default=str)


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print("Regulatrace Configuration:")
    print(settings.safe_repr())
