"""Offline checks for Step 5.1 HTTP contracts and dependency readiness."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from regulatrace.api import app as api_app
from regulatrace.api.schemas import ChatRequest

RESOURCE_ID = "00000000-0000-0000-0000-000000000001"
NOT_IMPLEMENTED = {
    "detail": {
        "code": "not_implemented",
        "message": "This workflow is not implemented in Step 5.1.",
    }
}


@pytest.fixture
def api_resources(monkeypatch):
    resources = SimpleNamespace(engine=Mock(), session=Mock(), qdrant=Mock())
    resources.session_context = MagicMock()
    resources.session_context.__enter__.return_value = resources.session
    resources.session_factory = Mock(return_value=resources.session_context)
    resources.settings = SimpleNamespace(
        app_name="Regulatrace",
        frontend_dist=None,
        database_null_pool=False,
        qdrant_url="http://qdrant.invalid:6333",
        qdrant_api_key=SecretStr("synthetic-qdrant-key"),
        cors_allowed_origins=["http://localhost:5173"],
    )
    resources.settings_factory = Mock(return_value=resources.settings)
    resources.engine_factory = Mock(return_value=resources.engine)
    resources.session_factory_factory = Mock(return_value=resources.session_factory)
    resources.qdrant_factory = Mock(return_value=resources.qdrant)
    monkeypatch.setattr(api_app, "get_settings", resources.settings_factory)
    monkeypatch.setattr(api_app, "get_engine", resources.engine_factory)
    monkeypatch.setattr(api_app, "get_session_factory", resources.session_factory_factory)
    monkeypatch.setattr(api_app, "QdrantClient", resources.qdrant_factory)
    resources.app = api_app.create_app()
    return resources


def test_openapi_contracts_do_not_initialize_resources(api_resources):
    schema = api_resources.app.openapi()
    contracts = {
        ("/auth/register", "post"): ("201", "UserResponse"),
        ("/auth/login", "post"): ("200", "TokenResponse"),
        ("/auth/refresh", "post"): ("200", "TokenResponse"),
        ("/chat", "post"): ("200", "ChatResponse"),
        ("/sessions", "get"): ("200", "SessionResponse"),
        ("/sessions/{id}", "get"): ("200", "SessionDetailResponse"),
        ("/feedback", "post"): ("201", "FeedbackResponse"),
        ("/documents", "get"): ("200", "DocumentResponse"),
        ("/documents/{id}/pdf", "get"): ("200", None),
        ("/admin/ingest", "post"): ("202", "JobResponse"),
        ("/admin/jobs/{id}", "get"): ("200", "JobResponse"),
        ("/health", "get"): ("200", "HealthResponse"),
        ("/ready", "get"): ("200", "ReadyResponse"),
    }
    assert {
        (path, method) for path, operations in schema["paths"].items() for method in operations
    } == set(contracts)
    for (path, method), (status, model) in contracts.items():
        responses = schema["paths"][path][method]["responses"]
        content = responses[status]["content"]
        if model is None:
            assert "application/pdf" in content
        else:
            response_schema = content["application/json"]["schema"]
            reference = {"$ref": f"#/components/schemas/{model}"}
            if path in {"/sessions", "/documents"}:
                assert response_schema["type"] == "array"
                assert response_schema["items"] == reference
            else:
                assert response_schema == reference
        if path not in {
            "/health",
            "/ready",
            "/chat",
            "/auth/register",
            "/auth/login",
            "/auth/refresh",
            "/sessions",
            "/sessions/{id}",
        }:
            assert responses["501"]["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/ErrorResponse"
            }

    ready_responses = schema["paths"]["/ready"]["get"]["responses"]
    assert ready_responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ReadyResponse"
    }
    # App creation reads settings for CORS, but must not initialize heavy resources.
    api_resources.settings_factory.assert_called_once_with()
    for factory in (
        api_resources.engine_factory,
        api_resources.session_factory_factory,
        api_resources.qdrant_factory,
    ):
        factory.assert_not_called()


def test_health_uses_lifespan_resources_without_probing_dependencies(api_resources):
    with TestClient(api_resources.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        api_resources.engine_factory.assert_called_once_with(
            connect_args={"connect_timeout": 3, "options": "-c statement_timeout=3000"},
            pool_timeout=3,
        )
        api_resources.session_factory_factory.assert_called_once_with(api_resources.engine)
        api_resources.qdrant_factory.assert_called_once_with(
            url="http://qdrant.invalid:6333",
            api_key="synthetic-qdrant-key",
            timeout=3,
            check_compatibility=False,
        )
        api_resources.session_factory.assert_not_called()
        api_resources.qdrant.get_collections.assert_not_called()
    api_resources.qdrant.close.assert_called_once_with()
    api_resources.engine.dispose.assert_called_once_with()


def test_ready_probes_postgresql_and_qdrant(api_resources):
    with TestClient(api_resources.app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {"postgresql": "ok", "qdrant": "ok"},
    }
    api_resources.session_context.__exit__.assert_called_once_with(None, None, None)
    api_resources.session.execute.assert_called_once()
    assert str(api_resources.session.execute.call_args.args[0]) == "SELECT 1"
    api_resources.qdrant.get_collections.assert_called_once_with()


def test_serverless_pooler_connection_omits_unsupported_startup_options(api_resources):
    api_resources.settings.database_null_pool = True
    with TestClient(api_resources.app) as client:
        assert client.get("/health").status_code == 200
    options = api_resources.engine_factory.call_args.kwargs["connect_args"]
    assert options["connect_timeout"] >= 10
    assert "options" not in options


@pytest.mark.parametrize("failed", [{"postgresql"}, {"qdrant"}, {"postgresql", "qdrant"}])
def test_ready_reports_each_failure_without_disclosing_errors(api_resources, failed):
    secret = "sentinel-password-and-api-key"
    if "postgresql" in failed:
        api_resources.session.execute.side_effect = RuntimeError(f"postgresql://user:{secret}@db")
    if "qdrant" in failed:
        api_resources.qdrant.get_collections.side_effect = RuntimeError(f"Authorization: {secret}")
    with TestClient(api_resources.app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {
            name: "unavailable" if name in failed else "ok" for name in ("postgresql", "qdrant")
        },
    }
    assert secret not in response.text
    assert "Traceback" not in response.text
    api_resources.session.execute.assert_called_once()
    api_resources.qdrant.get_collections.assert_called_once_with()


def test_deferred_workflow_routes_return_the_same_501_response(api_resources):
    requests = [
        ("POST", "/feedback", {"message_id": RESOURCE_ID, "rating": 1}),
        ("GET", "/documents", None),
        ("GET", f"/documents/{RESOURCE_ID}/pdf", None),
        ("POST", "/admin/ingest", {"source_file": "circular.pdf"}),
        ("GET", f"/admin/jobs/{RESOURCE_ID}", None),
    ]
    with TestClient(api_resources.app) as client:
        for method, path, body in requests:
            response = client.request(method, path, **({"json": body} if body is not None else {}))
            assert response.status_code == 501, path
            assert response.json() == NOT_IMPLEMENTED, path
    api_resources.session_factory.assert_not_called()
    api_resources.qdrant.get_collections.assert_not_called()


def test_requests_enforce_registration_and_question_contracts(api_resources):
    invalid_requests = [
        ("/auth/register", {"email": "analyst@example.com"}),
        (
            "/auth/register",
            {"email": "analyst@example.com", "password": "test-password", "role": "admin"},
        ),
    ]
    with TestClient(api_resources.app) as client:
        for path, body in invalid_requests:
            response = client.post(path, json=body)
            assert response.status_code == 422, (path, body)

    # Authentication runs before HTTP body validation; check the schema directly.
    with pytest.raises(ValidationError):
        ChatRequest(question=" \t\n")
