"""Model protocol, prepared evidence and free-account guards without cloud writes."""

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import UUID

import httpx
import pytest
from sqlalchemy.pool import NullPool

from regulatrace import remote_models
from regulatrace.cloud_corpus import load_prepared, publish_prepared
from regulatrace.config import Settings
from regulatrace.db.rate_limit import admit_chat_attempt
from regulatrace.model_worker import handle_model_request
from regulatrace.vectorstore import generate_point_id

KEY = "synthetic-worker-secret-01234567890123456789"
ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker_settings(monkeypatch):
    settings = Settings(_env_file=None, model_worker_key=KEY)
    monkeypatch.setattr(remote_models, "get_settings", lambda: settings)
    return settings


def test_api_imports_when_local_model_packages_are_absent():
    script = """
import sys,importlib.abc
blocked={'torch','transformers','sentence_transformers','fastembed','gradio','pymupdf'}
class ThinEnvironment(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in blocked:raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0,ThinEnvironment())
sys.path.insert(0,'src')
from regulatrace.api.app import create_app
from regulatrace.rag import build_rag_chain
assert '/chat' in create_app().openapi()['paths']
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=40
    )
    assert result.returncode == 0, result.stderr


def test_gradio_http_protocol_and_secret_placement(worker_settings):
    seen = []

    def respond(request):
        seen.append(request)
        assert KEY not in str(request.url)
        if request.method == "POST":
            assert json.loads(request.content)["data"][0]["key"] == KEY
            return httpx.Response(200, json={"event_id": "a123"})
        return httpx.Response(
            200,
            text=(
                "event: heartbeat\ndata: null\n\nevent: complete\n"
                'data: [{"protocol":1,"ready":true}]\n\n'
            ),
        )

    assert remote_models.call_worker({"action": "health"}, transport=httpx.MockTransport(respond))[
        "ready"
    ]
    assert [r.method for r in seen] == ["POST", "GET"]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_worker_provider_errors_do_not_expose_keys(worker_settings, status):
    transport = httpx.MockTransport(lambda r: httpx.Response(status, text=KEY))
    with pytest.raises(remote_models.ModelServiceUnavailable) as error:
        remote_models.call_worker({"action": "health"}, transport=transport)
    assert KEY not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://app.hf.space",
        "https://other.example.com",
        "https://user:pass@app.hf.space",
        "https://app.hf.space/?key=secret",
    ],
)
def test_worker_credentials_cannot_be_redirected(worker_settings, url):
    worker_settings.model_worker_url = url
    with pytest.raises(remote_models.ModelServiceUnavailable):
        remote_models.call_worker({"action": "health"})


def test_remote_ranking_preserves_canonical_evidence(monkeypatch):
    first = {"text": "first", "metadata": {"doc_id": "one", "page": 7}}
    second = {"text": "second", "metadata": {"doc_id": "two", "page": 8}}
    monkeypatch.setattr(
        remote_models,
        "call_worker",
        lambda p: {"model": remote_models.RERANK_MODEL, "scores": [0.2, 0.9]},
    )
    ranked = remote_models.remote_rerank("question", [first, second], 1)
    assert ranked == [{**second, "rerank_score": 0.9}]


@pytest.mark.parametrize("scores", [[True], [float("nan")], [], ["0.9"]])
def test_invalid_model_scores_fail_closed(monkeypatch, scores):
    monkeypatch.setattr(
        remote_models,
        "call_worker",
        lambda p: {"model": remote_models.RERANK_MODEL, "scores": scores},
    )
    with pytest.raises(remote_models.ModelServiceUnavailable):
        remote_models.remote_rerank("question", [{"text": "evidence"}])


def test_remote_embeddings_match_the_prepared_revision(monkeypatch):
    monkeypatch.setattr(
        remote_models,
        "call_worker",
        lambda p: {
            "model": remote_models.EMBEDDING_MODEL,
            "revision": remote_models.EMBEDDING_REVISION,
            "vectors": [[1.0] + [0.0] * 383 for _ in p["texts"]],
        },
    )
    values = remote_models.RemoteEmbedder().encode(["first", "second"])
    assert values.shape == (2, 384) and values[1, 0] == 1


def test_remote_embeddings_reject_different_weights(monkeypatch):
    monkeypatch.setattr(
        remote_models,
        "call_worker",
        lambda p: {
            "model": remote_models.EMBEDDING_MODEL,
            "revision": "different-revision",
            "vectors": [[1.0] + [0.0] * 383],
        },
    )
    with pytest.raises(remote_models.ModelServiceUnavailable):
        remote_models.RemoteEmbedder().encode("question")


def test_model_worker_checks_authentication_and_limits_before_gpu():
    rank, embed = Mock(), Mock()
    for payload in (
        {"key": "wrong", "action": "health"},
        {"key": KEY, "action": "rerank", "question": "q", "texts": ["a"] * 21},
    ):
        with pytest.raises(ValueError):
            handle_model_request(payload, secret=KEY, rank=rank, embed=embed)
    rank.assert_not_called()
    embed.assert_not_called()


def test_worker_scores_restore_input_order_and_health_needs_no_gpu():
    rank = Mock(return_value=[{"slot": 1, "rerank_score": 0.9}, {"slot": 0, "rerank_score": 0.2}])
    embed = Mock()
    health = handle_model_request(
        {"key": KEY, "action": "health"}, secret=KEY, rank=rank, embed=embed
    )
    assert health["ready"]
    rank.assert_not_called()
    result = handle_model_request(
        {"key": KEY, "action": "rerank", "question": "q", "texts": ["one", "two"]},
        secret=KEY,
        rank=rank,
        embed=embed,
    )
    assert result["scores"] == [0.2, 0.9]
    embed.assert_not_called()


def test_serverless_null_pool_discards_queue_pool_options(monkeypatch):
    from regulatrace.db import session

    monkeypatch.setattr(
        session, "get_settings", lambda: Settings(_env_file=None, database_null_pool=True)
    )
    engine = session.get_engine("sqlite://", pool_timeout=3, pool_size=4, max_overflow=1)
    assert isinstance(engine.pool, NullPool)
    engine.dispose()


def test_shared_limiter_rejects_an_eleventh_attempt_and_does_not_insert():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    session = MagicMock()
    session.execute.side_effect = [Mock(), Mock(), SimpleNamespace(one=lambda: (10, now))]
    assert admit_chat_attempt(session, UUID(int=1), now=now) == 60
    session.add.assert_not_called()


@pytest.fixture
def prepared(tmp_path):
    payload = {
        "doc_id": "document-id",
        "chunk_index": 1,
        "source_file": "source.pdf",
        "text": "Verified source text",
        "access_level": "public",
        "page": 1,
        "content_hash": hashlib.sha256(b"Verified source text").hexdigest(),
    }
    point = {
        "id": generate_point_id(payload["doc_id"], 1),
        "vector": [1.0] + [0.0] * 383,
        "payload": payload,
    }
    manifest = {
        "format": 1,
        "model": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "normalized": True,
        "point_count": 1,
        "document_count": 1,
        "documents": [{"source_file": "source.pdf", "chunks": 1}],
    }

    def save():
        with gzip.open(tmp_path / "points.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(point) + "\n")
        manifest["points_sha256"] = hashlib.sha256(
            (tmp_path / "points.jsonl.gz").read_bytes()
        ).hexdigest()
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    save()
    return tmp_path, point, manifest, save


def test_prepared_vector_and_page_metadata(prepared):
    folder, point, _, _ = prepared
    manifest, points = load_prepared(folder)
    assert manifest["point_count"] == 1 and points == [point]


@pytest.mark.parametrize("change", ["dimension", "normalization", "private", "page", "text", "id"])
def test_invalid_prepared_evidence_cannot_write_services(prepared, change):
    folder, point, _, save = prepared
    if change == "dimension":
        point["vector"].pop()
    elif change == "normalization":
        point["vector"][0] = 2
    elif change == "private":
        point["payload"]["access_level"] = "private"
    elif change == "page":
        point["payload"]["page"] = 0
    elif change == "text":
        point["payload"]["text"] = "modified evidence"
    else:
        point["id"] = generate_point_id("different-document", 1)
    save()
    client, sessions = Mock(), Mock()
    with pytest.raises(ValueError):
        publish_prepared(folder, client, sessions)
    assert not client.mock_calls and not sessions.mock_calls


def test_corrupted_vectors_cannot_reach_services(prepared):
    folder, _, _, _ = prepared
    with (folder / "points.jsonl.gz").open("ab") as stream:
        stream.write(b"corrupt")
    client = Mock()
    with pytest.raises(ValueError, match="checksum"):
        publish_prepared(folder, client, Mock())
    assert not client.mock_calls


def test_paid_vercel_account_cannot_create_resources(tmp_path, monkeypatch):
    driver = load_module("fullstack_paid_guard", "deploy/fullstack/deploy_fullstack.py")
    api = Mock(return_value={"teams": [{"id": "paid", "slug": "paid", "billing": {"plan": "pro"}}]})
    monkeypatch.setattr(driver, "vercel_api", api)
    with pytest.raises(driver.DeploymentError, match="free Hobby"):
        driver.select_hobby_team({}, tmp_path / "state.json")
    assert api.call_count == 1


def test_hobby_plan_is_rechecked_before_creating_resources(tmp_path, monkeypatch):
    driver = load_module("fullstack_plan_change", "deploy/fullstack/deploy_fullstack.py")
    api = Mock(
        side_effect=[
            {"teams": [{"id": "team", "slug": "slug", "billing": {"plan": "hobby"}}]},
            {"id": "team", "billing": {"plan": "pro"}},
        ]
    )
    monkeypatch.setattr(driver, "vercel_api", api)
    with pytest.raises(driver.DeploymentError, match="not on Hobby"):
        driver.select_hobby_team({}, tmp_path / "state.json")
    assert not (tmp_path / "state.json").exists()


def test_private_vercel_values_use_stdin_not_command_arguments(monkeypatch):
    driver = load_module("fullstack_stdin", "deploy/fullstack/deploy_fullstack.py")
    execute = Mock(return_value=SimpleNamespace(returncode=0, stdout="{}"))
    monkeypatch.setattr(driver.shutil, "which", lambda _: "npx")
    monkeypatch.setattr(driver.subprocess, "run", execute)
    driver.cli(["api", "/endpoint", "--input", "-"], payload="synthetic-private-value")
    assert "synthetic-private-value" not in str(execute.call_args.args)
    assert execute.call_args.kwargs["input"] == "synthetic-private-value"


@pytest.mark.parametrize("npm_location", ["npx", "node"])
@pytest.mark.parametrize("interactive", [False, True])
def test_windows_vercel_uses_native_node_with_literal_url_and_stdin(
    tmp_path, monkeypatch, npm_location, interactive
):
    driver = load_module("fullstack_windows_npm", "deploy/fullstack/deploy_fullstack.py")
    npx = tmp_path / "npm wrappers & spaces" / "npx.cmd"
    node = tmp_path / "node runtime & spaces" / "node.exe"
    entry = (npx if npm_location == "npx" else node).parent / "node_modules/npm/bin/npx-cli.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// synthetic npm entry point")
    monkeypatch.setattr(driver, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        driver.shutil, "which", lambda name: {"npx.cmd": str(npx), "node.exe": str(node)}.get(name)
    )
    execute = Mock(return_value=SimpleNamespace(returncode=0, stdout="{}"))
    monkeypatch.setattr(driver.subprocess, "run", execute)
    endpoint = "/v10/projects/prj_owned/env?upsert=true&teamId=team_owned"
    driver.cli(
        ["api", endpoint, "--method", "POST", "--input", "-"],
        payload="synthetic-private-value",
        interactive=interactive,
    )
    arguments = execute.call_args.args[0]
    assert arguments[:2] == [str(node), str(entry)]
    assert str(npx) not in arguments
    assert endpoint in arguments and "synthetic-private-value" not in arguments
    assert execute.call_args.kwargs.get("shell", False) is False
    if not interactive:
        assert execute.call_args.kwargs["input"] == "synthetic-private-value"


@pytest.mark.parametrize("missing", ["node", "entry"])
def test_windows_missing_node_launcher_stops_before_any_cli_call(tmp_path, monkeypatch, missing):
    driver = load_module("fullstack_windows_missing_npm", "deploy/fullstack/deploy_fullstack.py")
    npx = tmp_path / "npx.cmd"
    monkeypatch.setattr(driver, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        driver.shutil,
        "which",
        lambda name: (
            str(npx)
            if name == "npx.cmd"
            else (str(tmp_path / "node.exe") if missing != "node" else None)
        ),
    )
    execute = Mock()
    monkeypatch.setattr(driver.subprocess, "run", execute)
    with pytest.raises(driver.DeploymentError, match="Node.js"):
        driver.cli(["api", "/v2/teams"])
    execute.assert_not_called()


def test_unrelated_neon_database_is_rejected():
    driver = load_module("fullstack_database_guard", "deploy/fullstack/deploy_fullstack.py")
    with pytest.raises(driver.DeploymentError, match="dedicated Neon"):
        driver.validate_config(
            {
                "REGULATRACE_DATABASE_URL": "postgresql://user:pass@host.neon.tech/news_verifier?sslmode=require"
            }
        )


@pytest.mark.parametrize("created_as_list", [False, True])
def test_vercel_settings_use_object_bodies_and_production_secret_types(
    monkeypatch, capsys, created_as_list
):
    driver = load_module("fullstack_environment_objects", "deploy/fullstack/deploy_fullstack.py")
    requests = []

    def api(endpoint, *, method, body, team):
        assert isinstance(body, dict)
        assert endpoint == "/v10/projects/prj_owned/env?upsert=true"
        assert method == "POST" and team == {"id": "team_owned"}
        requests.append(body)
        created = {"key": body["key"], "target": body["target"]}
        return {"created": [created] if created_as_list else created, "failed": []}

    monkeypatch.setattr(driver, "vercel_api", api)
    values = {
        "REGULATRACE_DATABASE_URL": "synthetic-database-private-sentinel",
        "REGULATRACE_MODEL_WORKER_SECRET": "synthetic-worker-private-sentinel",
        "REGULATRACE_GROQ_API_KEY": "synthetic-groq-private-sentinel",
        "REGULATRACE_ENVIRONMENT": "production",
        "OPTIONAL_EMPTY": "",
    }
    driver.set_environment({"id": "prj_owned"}, values, {"id": "team_owned"})
    assert len(requests) == 4
    assert all(body["target"] == ["production"] for body in requests)
    assert [body["type"] for body in requests] == ["sensitive"] * 3 + ["plain"]
    assert all(body["value"] == values[body["key"]] for body in requests)
    output = capsys.readouterr().out
    assert all(secret not in output for secret in list(values.values())[:3])


@pytest.mark.parametrize("failure", ["missing", "mismatched", "wrong_target", "rejected"])
def test_vercel_setting_failures_stop_without_exposing_values(monkeypatch, capsys, failure):
    driver = load_module("fullstack_environment_errors", "deploy/fullstack/deploy_fullstack.py")
    value = "synthetic-private-setting-sentinel"
    created = {"key": "REGULATRACE_GROQ_API_KEY", "target": ["production"]}
    result = {"created": created, "failed": []}
    if failure == "missing":
        result["created"] = None
    elif failure == "mismatched":
        created["key"] = "DIFFERENT_KEY"
    elif failure == "wrong_target":
        created["target"] = ["preview"]
    else:
        result["failed"] = [{"value": value, "error": {"message": value}}]
    monkeypatch.setattr(driver, "vercel_api", Mock(return_value=result))
    with pytest.raises(driver.DeploymentError) as error:
        driver.set_environment({"id": "prj_owned"}, {"REGULATRACE_GROQ_API_KEY": value}, {})
    assert value not in str(error.value) and value not in capsys.readouterr().out


@pytest.mark.parametrize("publish_fails", [False, True])
def test_launcher_closes_real_qdrant_client_after_publication(tmp_path, monkeypatch, publish_fails):
    import qdrant_client

    driver = load_module("fullstack_qdrant_cleanup", "deploy/fullstack/deploy_fullstack.py")
    # main updates these names; track them so this fixture cannot change later tests.
    monkeypatch.setenv("REGULATRACE_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("REGULATRACE_QDRANT_API_KEY", "")
    local_client = qdrant_client.QdrantClient(":memory:")
    engine = Mock()
    publish = Mock(side_effect=ValueError("synthetic publish failure") if publish_fails else None)
    publish.return_value = {"documents": 42}

    def configure(folder):
        folder.mkdir(parents=True)
        return {
            "REGULATRACE_QDRANT_URL": "https://unused.cloud.qdrant.io",
            "REGULATRACE_QDRANT_API_KEY": "synthetic-key",
        }

    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["deploy_fullstack.py"])
    monkeypatch.setattr(driver, "configure", configure)
    monkeypatch.setattr(driver, "load_prepared", lambda _: ({}, []))
    monkeypatch.setattr(driver, "select_hobby_team", Mock(return_value={"slug": "test"}))
    monkeypatch.setattr(driver, "get_engine", Mock(return_value=engine))
    monkeypatch.setattr(driver, "migrate", Mock())
    monkeypatch.setattr(driver, "sessionmaker", Mock(return_value=Mock()))
    monkeypatch.setattr(qdrant_client, "QdrantClient", Mock(return_value=local_client))
    monkeypatch.setattr(driver, "publish_prepared", publish)
    model_space = Mock(return_value="synthetic-commit")
    monkeypatch.setattr(driver, "update_model_space", model_space)
    monkeypatch.setattr(driver, "project", Mock(return_value={"id": "synthetic"}))
    monkeypatch.setattr(driver, "set_environment", Mock())
    monkeypatch.setattr(driver, "deploy", Mock(return_value="https://synthetic.vercel.app"))
    monkeypatch.setattr(driver, "smoke_test", Mock(return_value={"synthetic": "passed"}))

    assert driver.main() == (1 if publish_fails else 0)
    publish.assert_called_once()
    assert publish.call_args.args[1] is local_client
    assert local_client._client.closed
    engine.dispose.assert_called_once()
    assert model_space.called is not publish_fails
    assert (tmp_path / "DEPLOYMENT_RESULT.json").exists() is not publish_fails


def test_hosted_bundles_exclude_private_files_and_heavy_api_dependencies(tmp_path):
    builder = load_module("fullstack_bundler", "deploy/fullstack/build_bundle.py")
    builder.build(tmp_path / "bundles")
    files = [p for p in (tmp_path / "bundles").rglob("*") if p.is_file()]
    assert not any(
        p.name.startswith(".env") or p.suffix.lower() in {".pdf", ".safetensors"} for p in files
    )
    import tomllib

    requirements = tomllib.loads((tmp_path / "bundles/api/pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    names = {r.split("==")[0].split("[")[0].lower() for r in requirements}
    assert not names & {
        "torch",
        "transformers",
        "gradio",
        "fastembed",
        "sentence-transformers",
        "pymupdf",
    }
    assert {"fastapi", "groq", "psycopg2-binary", "qdrant-client"} <= names
