"""Unit tests for Regulatrace configuration."""

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from regulatrace.config import Settings, get_settings


def test_groq_credentials_default_to_none():
    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).groq_api_key is None


def test_groq_settings_load_synthetic_environment_and_mask_secret():
    test_key = "synthetic-groq-test-key"
    with patch.dict(
        os.environ,
        {
            "REGULATRACE_LLM_PROVIDER": "groq",
            "REGULATRACE_LLM_MODEL": "configured-groq-model",
            "REGULATRACE_GROQ_API_KEY": test_key,
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "configured-groq-model"
    assert isinstance(settings.groq_api_key, SecretStr)
    assert settings.groq_api_key.get_secret_value() == test_key
    assert settings.safe_dump()["groq_api_key"] == "**********"
    assert test_key not in settings.safe_repr()
    assert test_key not in repr(settings)


def test_gemini_ipv4_default():
    """The workaround is opt-in and independent of the developer's .env."""
    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).gemini_force_ipv4 is False


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_gemini_ipv4_environment_override(value, expected):
    with patch.dict(os.environ, {"REGULATRACE_GEMINI_FORCE_IPV4": value}, clear=True):
        assert Settings(_env_file=None).gemini_force_ipv4 is expected


def test_default_settings():
    """Verify default settings values when no environment variables are set."""
    # Ensure cache does not return an altered instance
    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        assert settings.app_name == "Regulatrace"
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.debug is False


def test_environment_variable_override():
    """Verify settings can be overridden via REGULATRACE_ prefixed environment variables."""
    get_settings.cache_clear()
    custom_env = {
        "REGULATRACE_APP_NAME": "CustomRegulatrace",
        "REGULATRACE_ENVIRONMENT": "production",
        "REGULATRACE_LOG_LEVEL": "WARNING",
        "REGULATRACE_DEBUG": "true",
    }
    with patch.dict(os.environ, custom_env, clear=True):
        settings = Settings()
        assert settings.app_name == "CustomRegulatrace"
        assert settings.environment == "production"
        assert settings.log_level == "WARNING"
        assert settings.debug is True


def test_safe_dump_masks_sensitive_keys():
    """Verify safe_dump masks any field whose name contains sensitive keywords."""
    settings = Settings()
    from pydantic import SecretStr

    settings = Settings(
        groq_api_key=SecretStr("super-secret-key-123"),
        gemini_api_key=SecretStr("gemini-secret-token-456"),
    )
    safe = settings.safe_dump()
    assert "app_name" in safe
    assert safe["app_name"] == "Regulatrace"
    # Ensure safe_repr is valid JSON
    assert safe["groq_api_key"] == "**********"
    assert safe["gemini_api_key"] == "**********"
    # Ensure safe_repr is valid JSON and does not expose the secrets
    repr_str = settings.safe_repr()
    assert '"app_name": "Regulatrace"' in repr_str
    assert "super-secret-key-123" not in repr_str
    assert "gemini-secret-token-456" not in repr_str


def test_database_url_password_is_masked_without_changing_connection_value():
    url = "postgresql+psycopg://analyst:p%40ss%3Aword@db.example:5432/regulatrace?sslmode=require"
    settings = Settings(_env_file=None, database_url=url)

    safe_url = settings.safe_dump()["database_url"]
    assert settings.database_url == url
    assert "db.example:5432/regulatrace?sslmode=require" in safe_url
    assert "analyst" in safe_url
    assert "p%40ss%3Aword" not in safe_url
    assert "p@ss:word" not in safe_url
    assert "p%40ss%3Aword" not in settings.safe_repr()


def test_unparseable_database_url_is_fully_masked():
    settings = Settings(_env_file=None, database_url="malformed-secret-value")
    assert settings.safe_dump()["database_url"] == "**********"
    assert "malformed-secret-value" not in settings.safe_repr()


def test_database_url_query_credentials_are_masked():
    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql://db.example/regulatrace?sslmode=require&sslpassword=query-secret"
        ),
    )
    safe_url = settings.safe_dump()["database_url"]
    assert "sslmode=require" in safe_url
    assert "query-secret" not in safe_url
    assert "query-secret" not in settings.safe_repr()


def test_real_environment_precedes_dotenv_and_cached_settings_need_refresh(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "REGULATRACE_LLM_PROVIDER=groq\nREGULATRACE_LLM_MODEL=llama-3.3-70b-versatile\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REGULATRACE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("REGULATRACE_LLM_MODEL", "gemini-test-model")
    get_settings.cache_clear()
    try:
        first = get_settings()
        assert (first.llm_provider, first.llm_model) == ("gemini", "gemini-test-model")
        monkeypatch.delenv("REGULATRACE_LLM_PROVIDER")
        monkeypatch.delenv("REGULATRACE_LLM_MODEL")
        assert get_settings() is first
        get_settings.cache_clear()
        refreshed = get_settings()
        assert (refreshed.llm_provider, refreshed.llm_model) == (
            "groq",
            "llama-3.3-70b-versatile",
        )
    finally:
        get_settings.cache_clear()
