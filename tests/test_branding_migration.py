"""Existing credentials, evidence identifiers and sessions survive a branding migration."""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import jwt
import pytest

from regulatrace.api import auth
from regulatrace.config import Settings

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "branding_migration", ROOT / "deploy/fullstack/migrate_branding.py"
)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)
KEY = bytes(range(32)).hex()


@pytest.fixture
def saved_setup(tmp_path):
    private = tmp_path / ".fullstack-state"
    private.mkdir()
    settings = {
        "EARLIERAPP_DATABASE_URL": "postgresql://user:earlierapp-password@host/regulatrace",
        "EARLIERAPP_QDRANT_URL": "https://existing.cloud.qdrant.io",
        "EARLIERAPP_QDRANT_API_KEY": "synthetic-vector-secret",
        "EARLIERAPP_JWT_SECRET": KEY,
        "REGULATRACE_GROQ_API_KEY": "synthetic-groq-secret",
        "REGULATRACE_MODEL_WORKER_SECRET": "synthetic-worker-secret",
    }
    state = {
        "api_project": {"id": "api-owned"},
        "web_project": {"id": "web-owned"},
        "corpus": {"dense_collection": "regulatrace_existing_dense", "dense_points": 672},
    }
    (private / "secrets.json").write_text(json.dumps(settings))
    (private / "deployment.json").write_text(json.dumps(state))
    return tmp_path, private, settings, state


def test_private_namespace_migration_preserves_values_indexes_and_stable_backup(
    saved_setup, monkeypatch
):
    package, private, original, original_state = saved_setup
    raw = (private / "secrets.json").read_bytes()
    monkeypatch.setattr(migration.time, "time", lambda: 1000)
    assert migration.migrate_settings(package, "earlierapp")["settings_migrated"] == 4
    changed = json.loads((private / "secrets.json").read_text())
    for old_key, value in original.items():
        assert changed[old_key.replace("EARLIERAPP_", "REGULATRACE_", 1)] == value
    state = json.loads((private / "deployment.json").read_text())
    assert all(state[key] == value for key, value in original_state.items())
    assert (private / "branding-backup/secrets.json").read_bytes() == raw
    deadline = changed["REGULATRACE_PREVIOUS_TOKEN_VALID_UNTIL"]
    monkeypatch.setattr(migration.time, "time", lambda: 2000)
    assert migration.migrate_settings(package, "regulatrace")["settings_migrated"] == 0
    assert json.loads((private / "secrets.json").read_text()) == changed
    assert deadline == str(1000 + 7 * 86400)
    assert (private / "branding-backup/secrets.json").read_bytes() == raw


@pytest.mark.parametrize("failure", ["conflict", "missing", "unsafe_name"])
def test_unsafe_private_migrations_leave_settings_unchanged(saved_setup, failure):
    package, private, settings, _ = saved_setup
    if failure == "conflict":
        settings["REGULATRACE_JWT_SECRET"] = "different-secret"
    elif failure == "missing":
        del settings["EARLIERAPP_DATABASE_URL"]
    (private / "secrets.json").write_text(json.dumps(settings))
    before = (private / "secrets.json").read_bytes()
    with pytest.raises(ValueError) as error:
        migration.migrate_settings(
            package, "../outside" if failure == "unsafe_name" else "earlierapp"
        )
    assert (private / "secrets.json").read_bytes() == before
    assert "different-secret" not in str(error.value)
    assert not (private / "branding-backup").exists()


def test_local_configuration_keys_change_without_editing_private_values(saved_setup):
    package, private, _, _ = saved_setup
    migration.migrate_settings(package, "earlierapp")
    repository = package / "source"
    repository.mkdir()
    original = 'EARLIERAPP_DATABASE_URL="postgresql://u:earlierapp-password@host/db?a=1&b=2"\n'
    (repository / ".env").write_text(original)
    assert migration.migrate_dotenv(repository, package)
    assert (repository / ".env").read_text() == original.replace(
        "EARLIERAPP_DATABASE_URL", "REGULATRACE_DATABASE_URL"
    )
    assert (private / "branding-backup/repository.env").read_text() == original
    assert not migration.migrate_dotenv(repository, package)


def test_only_retired_bundle_modules_are_removed_and_preserved_privately(saved_setup):
    package, private, _, _ = saved_setup
    migration.migrate_settings(package, "earlierapp")
    for kind in ("api", "HuggingFace_Space"):
        for name in ("earlierapp", "regulatrace", "other_module"):
            folder = package / "bundles" / kind / "src" / name
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text(name)
    migration.remove_retired_bundle_modules(package)
    for kind in ("api", "HuggingFace_Space"):
        assert not (package / "bundles" / kind / "src/earlierapp").exists()
        assert (package / "bundles" / kind / "src/regulatrace/__init__.py").is_file()
        assert (package / "bundles" / kind / "src/other_module/__init__.py").is_file()
        backup = private / "branding-backup/bundles" / kind / "src/earlierapp/__init__.py"
        assert backup.read_text() == "earlierapp"


@pytest.fixture
def token_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        jwt_secret=KEY,
        previous_token_issuer="earlierapp",
        previous_token_valid_until=int(datetime.now(timezone.utc).timestamp()) + 1000,
    )
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(
        auth, "find_user_by_id", lambda session, user_id: SimpleNamespace(role="user")
    )
    monkeypatch.setattr(auth, "record_user_role", lambda role: None)
    return settings, MagicMock(), uuid4()


def signed_token(user_id, kind, issuer="earlierapp", audience="earlierapp-api", expired=False):
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user_id),
            "type": kind,
            "iss": issuer,
            "aud": audience,
            "iat": now - timedelta(seconds=10),
            "jti": str(uuid4()),
            "exp": now + timedelta(seconds=-1 if expired else 100),
        },
        bytes.fromhex(KEY),
        algorithm="HS256",
    )


@pytest.mark.parametrize("kind", ["access", "refresh"])
def test_new_tokens_use_regulatrace_and_existing_tokens_keep_their_account(token_settings, kind):
    _, session, user_id = token_settings
    token = auth._create_token(user_id, kind)
    claims = jwt.decode(
        token,
        bytes.fromhex(KEY),
        algorithms=["HS256"],
        issuer="regulatrace",
        audience="regulatrace-api",
    )
    assert claims["sub"] == str(user_id)
    assert auth.token_user_id(session, signed_token(user_id, kind), kind) == user_id


@pytest.mark.parametrize("failure", ["cutoff", "disabled", "expired", "mixed", "wrong_type"])
def test_transition_rejects_expired_and_unrelated_tokens(token_settings, failure):
    settings, session, user_id = token_settings
    if failure == "cutoff":
        settings.previous_token_valid_until = 1
    elif failure == "disabled":
        settings.previous_token_issuer = None
    token = signed_token(
        user_id,
        "refresh" if failure == "wrong_type" else "access",
        audience="regulatrace-api" if failure == "mixed" else "earlierapp-api",
        expired=failure == "expired",
    )
    with pytest.raises(auth.AuthError) as error:
        auth.token_user_id(session, token, "access")
    assert error.value.status_code == 401
    assert token not in str(error.value)


def test_existing_refresh_expiry_is_not_extended_during_transition(token_settings):
    _, session, user_id = token_settings
    previous = signed_token(user_id, "refresh")
    refreshed = auth.refresh_tokens(session, previous)
    assert refreshed.refresh_token == previous
    claims = jwt.decode(
        refreshed.access_token,
        bytes.fromhex(KEY),
        algorithms=["HS256"],
        issuer="regulatrace",
        audience="regulatrace-api",
    )
    assert claims["sub"] == str(user_id)
