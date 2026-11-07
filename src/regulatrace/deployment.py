"""Credential-safe deployment checks; no inference or writes are performed."""

import argparse
from contextlib import suppress
from urllib.parse import urlsplit

from qdrant_client import QdrantClient, models
from sqlalchemy import text
from sqlalchemy.exc import ArgumentError

from regulatrace.config import Settings, get_settings
from regulatrace.db.session import get_engine, normalize_database_url

SCHEMA_REVISION = "d18e42c63b90"


def configuration_checks(settings: Settings) -> dict[str, bool]:
    """Validate the external-service production profile without printing values."""
    try:
        signing_key = bytes.fromhex(
            settings.jwt_secret.get_secret_value() if settings.jwt_secret else ""
        )
    except ValueError:
        signing_key = b""

    def https_origin(value: str) -> bool:
        try:
            url = urlsplit(value)
            return bool(
                url.scheme == "https"
                and url.hostname
                and not url.username
                and not url.password
                and url.path in {"", "/"}
                and not url.query
                and not url.fragment
                and "*" not in value
            )
        except ValueError:
            return False

    try:
        database = normalize_database_url(settings.database_url)
        database_ok = bool(
            database.drivername == "postgresql+psycopg2"
            and database.host
            and database.database
            and database.username
            and database.password
            and database.query.get("sslmode") in {"require", "verify-ca", "verify-full"}
        )
    except (ArgumentError, ValueError, TypeError):
        database_ok = False

    return {
        "production_environment": settings.environment == "production" and not settings.debug,
        "jwt_signing_key": len(signing_key) == 32 and len(set(signing_key)) >= 16,
        "groq_provider": settings.llm_provider.strip().lower() == "groq",
        "groq_key": bool(
            settings.groq_api_key and settings.groq_api_key.get_secret_value().strip()
        ),
        "generation_model": bool(settings.llm_model.strip()),
        "postgresql_tls_url": database_ok,
        "qdrant_https_url": https_origin(settings.qdrant_url),
        "qdrant_key": bool(
            settings.qdrant_api_key and settings.qdrant_api_key.get_secret_value().strip()
        ),
        "separate_indexes": bool(
            settings.qdrant_collection.strip()
            and settings.bm25_collection.strip()
            and settings.qdrant_collection != settings.bm25_collection
        ),
        # An empty list is valid for the bundled same-origin frontend.
        "cors_origins": all(https_origin(origin) for origin in settings.cors_allowed_origins),
        "frontend_build": settings.frontend_dist is None
        or (settings.frontend_dist / "index.html").is_file(),
    }


def service_checks(settings: Settings) -> dict[str, bool]:
    """Read database/index/model metadata without sending a generation request."""
    checks = {
        "postgresql_schema": False,
        "dense_and_bm25_counts": False,
        "public_bm25_evidence": False,
        "groq_model_available": False,
    }
    engine = None
    try:
        engine = get_engine(
            settings.database_url, connect_args={"connect_timeout": 5}, pool_timeout=5
        )
        with engine.connect() as connection:
            revisions = set(connection.execute(text("SELECT version_num FROM alembic_version")))
            checks["postgresql_schema"] = (SCHEMA_REVISION,) in revisions
    except Exception:
        pass
    finally:
        if engine is not None:
            with suppress(Exception):
                engine.dispose()

    client = None
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
            timeout=5,
            check_compatibility=False,
        )
        dense = client.count(settings.qdrant_collection, exact=True).count
        sparse = client.count(settings.bm25_collection, exact=True).count
        checks["dense_and_bm25_counts"] = dense > 0 and dense == sparse
        public_filter = models.Filter(
            must=[
                models.FieldCondition(key="access_level", match=models.MatchValue(value="public"))
            ]
        )
        public_count = client.count(
            settings.bm25_collection, count_filter=public_filter, exact=True
        ).count
        sparse_config = client.get_collection(settings.bm25_collection).config.params.sparse_vectors
        checks["public_bm25_evidence"] = bool(
            public_count > 0
            and sparse_config
            and "bm25" in sparse_config
            and sparse_config["bm25"].modifier == models.Modifier.IDF
        )
    except Exception:
        pass
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()

    try:
        from groq import Groq

        with Groq(
            api_key=settings.groq_api_key.get_secret_value(), timeout=5, max_retries=0
        ) as provider:
            checks["groq_model_available"] = any(
                model.id == settings.llm_model for model in provider.models.list().data
            )
    except Exception:
        pass
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Regulatrace deployment configuration")
    parser.add_argument("--live", action="store_true", help="Also read service metadata")
    args = parser.parse_args()
    try:
        settings = get_settings()
        checks = configuration_checks(settings)
    except Exception:
        print("FAIL: settings could not be loaded; no configuration values are printed.")
        return 1
    if all(checks.values()) and args.live:
        checks.update(service_checks(settings))
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if all(checks.values()):
        print("Checks passed. Authenticated chat and citation smoke tests are still required.")
        return 0
    print("Deployment checks failed. Configure the failed items before release.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
