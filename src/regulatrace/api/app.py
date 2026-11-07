"""Create the API without connecting to services or loading models at import time."""

from contextlib import asynccontextmanager
from logging import getLogger
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from starlette.concurrency import run_in_threadpool

from regulatrace.api.auth import AuthError
from regulatrace.api.routes import router
from regulatrace.api.schemas import ChatErrorResponse, ErrorDetail
from regulatrace.api.services import ChatServiceError
from regulatrace.config import get_settings
from regulatrace.db.repository import save_query_log
from regulatrace.db.session import get_engine, get_session_factory
from regulatrace.tracing import chat_trace, completed_query_log, update_observation


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # PgBouncer rejects session startup options such as statement_timeout.
    connect_args = (
        {"connect_timeout": 10}
        if settings.database_null_pool
        else {"connect_timeout": 3, "options": "-c statement_timeout=3000"}
    )
    engine = get_engine(connect_args=connect_args, pool_timeout=3)
    try:
        key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = QdrantClient(
            url=settings.qdrant_url, api_key=key, timeout=3, check_compatibility=False
        )
        try:
            app.state.session_factory = get_session_factory(engine)
            app.state.qdrant_client = client
            yield
        finally:
            client.close()
    finally:
        engine.dispose()


def create_app() -> FastAPI:
    """Register authentication, JSON chat, health and deferred workflow contracts."""
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version="0.1.0",
        description=(
            "Authentication, JSON /chat, sessions, health and readiness are implemented. "
            "Chat requires a bearer access token and uses public documents only. "
            "Session memory is bounded and owner-scoped. Streaming is deferred; "
            "other operations return HTTP 501."
        ),
        lifespan=lifespan,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
    )
    # Single-process state shared by concurrent request threads, reset with the app.
    app.state.chat_request_times = {}
    app.state.chat_rate_lock = Lock()

    @app.middleware("http")
    async def trace_chat_request(request: Request, call_next):
        if request.method != "POST" or request.url.path != "/chat":
            return await call_next(request)
        with chat_trace() as observation:
            response = await call_next(request)
            update_observation(
                observation,
                metadata={"http_status": response.status_code},
                level="ERROR"
                if response.status_code >= 500
                else ("WARNING" if response.status_code >= 400 else "DEFAULT"),
            )
            if response.status_code < 400:
                try:
                    values = completed_query_log()
                    if values is not None:
                        await run_in_threadpool(save_query_log, app.state.session_factory, values)
                except Exception:
                    # Logging failure must not undo a saved chat or leak error details.
                    getLogger(__name__).warning("Completed chat query log could not be saved.")
            return response

    @app.exception_handler(AuthError)
    async def auth_error(request: Request, exc: AuthError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": str(exc)}},
            headers={"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None,
        )

    @app.exception_handler(ChatServiceError)
    async def chat_error(request: Request, exc: ChatServiceError):
        error = ChatErrorResponse(
            detail=ErrorDetail(code=exc.code, message=str(exc)), sources=exc.sources
        )
        return JSONResponse(
            status_code={"session_not_found": 404, "chat_failed": 500}.get(exc.code, 503),
            content=error.model_dump(mode="json"),
        )

    # Add last so CORS covers responses from the existing request middleware too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Retry-After", "WWW-Authenticate"],
    )
    app.include_router(router)
    if settings.frontend_dist is not None:
        if not (settings.frontend_dist / "index.html").is_file():
            raise RuntimeError("Configured frontend build is missing index.html.")
        # Registered after API routes so /auth, /chat and /health keep their contracts.
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    return app
