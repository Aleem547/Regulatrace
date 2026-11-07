"""Deployment regressions independent of database servers, keys and model weights."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from regulatrace.api import app as api_app
from regulatrace.config import Settings
from regulatrace.db.session import get_engine, normalize_database_url
from regulatrace.deployment import configuration_checks
from regulatrace.prepare_index import prepare_index, validate_corpus


@pytest.mark.parametrize("scheme", ["postgres", "postgresql", "postgresql+psycopg2"])
def test_generic_database_urls_use_installed_driver_without_losing_password(scheme):
    value = f"{scheme}://user:p%40ss%2525@localhost:5432/db?sslmode=require"
    normalized = normalize_database_url(value)
    assert normalized.drivername == "postgresql+psycopg2"
    assert normalized.password == "p@ss%25"
    engine = get_engine(value)
    try:
        assert engine.dialect.driver == "psycopg2"
        assert engine.url.query["sslmode"] == "require"
    finally:
        engine.dispose()


def test_sqlite_engine_still_supports_existing_offline_callers():
    engine = get_engine("sqlite://")
    try:
        assert engine.dialect.name == "sqlite"
    finally:
        engine.dispose()


def production_settings(**changes):
    values = {
        "environment": "production",
        "jwt_secret": bytes(range(32)).hex(),
        "llm_provider": "groq",
        "groq_api_key": "synthetic-private-groq-key",
        "llm_model": "synthetic-model",
        "database_url": "postgresql://user:synthetic-password@db.test/db?sslmode=require",
        "qdrant_url": "https://qdrant.test",
        "qdrant_api_key": "synthetic-private-qdrant-key",
        "cors_allowed_origins": [],
    }
    return Settings(_env_file=None, **(values | changes))


def test_bundled_same_origin_configuration_needs_no_cross_origin_allowlist():
    assert all(configuration_checks(production_settings()).values())


@pytest.mark.parametrize(
    "change, failed",
    [
        ({"jwt_secret": "00" * 32}, "jwt_signing_key"),
        ({"llm_provider": "mock"}, "groq_provider"),
        ({"groq_api_key": None}, "groq_key"),
        ({"cors_allowed_origins": ["*"]}, "cors_origins"),
        ({"database_url": "postgresql://user:password@db.test/db"}, "postgresql_tls_url"),
        ({"database_url": "invalid database URL"}, "postgresql_tls_url"),
        ({"qdrant_url": "http://qdrant.test"}, "qdrant_https_url"),
        ({"qdrant_collection": "same", "bm25_collection": "same"}, "separate_indexes"),
    ],
)
def test_unsafe_production_inputs_fail_without_putting_secrets_in_results(change, failed):
    checks = configuration_checks(production_settings(**change))
    assert checks[failed] is False
    assert "synthetic-private" not in repr(checks)
    assert "synthetic-password" not in repr(checks)


def test_frontend_and_api_are_served_on_one_origin(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<h1>Regulatrace</h1>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("window.example = true;")
    settings = production_settings(frontend_dist=tmp_path)
    monkeypatch.setattr(api_app, "get_settings", lambda: settings)
    application = api_app.create_app()
    # No lifespan here: the test must not create a real PostgreSQL/Qdrant connection.
    client = TestClient(application)
    assert "Regulatrace" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    assert "/auth/login" in client.get("/openapi.json").json()["paths"]
    assert client.get("/outside-secret.env").status_code == 404


def test_missing_frontend_build_is_rejected_instead_of_deploying_an_empty_page(
    tmp_path, monkeypatch
):
    settings = production_settings(frontend_dist=tmp_path)
    assert configuration_checks(settings)["frontend_build"] is False
    monkeypatch.setattr(api_app, "get_settings", lambda: settings)
    with pytest.raises(RuntimeError, match="frontend build"):
        api_app.create_app()


def test_incomplete_corpus_is_rejected_before_any_remote_mutation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text('file,date\nmissing.pdf,"Jan 01, 2026"\n')
    factory = Mock()
    monkeypatch.setattr("regulatrace.prepare_index.QdrantClient", factory)
    with pytest.raises(ValueError, match="Missing 1 of 1"):
        prepare_index(tmp_path, manifest)
    factory.assert_not_called()


def test_html_disguised_as_a_regulatory_pdf_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text('file,date\nnotice.pdf,"Jan 01, 2026"\n')
    (tmp_path / "notice.pdf").write_text("<html>Access denied</html>")
    with pytest.raises(ValueError, match="not a PDF"):
        validate_corpus(tmp_path, manifest)
