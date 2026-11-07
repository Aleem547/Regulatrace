"""Access lifespan resources without constructing clients for each request."""

from collections import deque
from collections.abc import Generator
from math import ceil
from time import monotonic
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from qdrant_client import QdrantClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from regulatrace.api.auth import AuthError, token_user_id
from regulatrace.config import get_settings
from regulatrace.db.rate_limit import admit_chat_attempt

_bearer = HTTPBearer(auto_error=False)
_CHAT_REQUEST_LIMIT = 10
_CHAT_WINDOW_SECONDS = 60


def get_session(request: Request) -> Generator[Session, None, None]:
    with request.app.state.session_factory() as session:
        yield session


def get_qdrant(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: Annotated[Session, Depends(get_session)],
) -> UUID:
    """Resolve identity from a verified access token, never from request-body fields."""
    if credentials is None:
        raise AuthError(401, "authentication_required", "A bearer access token is required.")
    return token_user_id(session, credentials.credentials, "access")


def get_rate_limited_chat_user_id(
    request: Request,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session: Annotated[Session, Depends(get_session)],
) -> UUID:
    """Admit at most ten chat attempts per verified user in a rolling minute."""
    if get_settings().chat_rate_limit_store == "postgres":
        try:
            retry_after = admit_chat_attempt(session, user_id)
        except SQLAlchemyError:
            raise AuthError(
                503, "rate_limit_unavailable", "Chat admission is unavailable."
            ) from None
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "rate_limit_exceeded",
                    "message": "Chat is limited to 10 requests per 60 seconds.",
                },
                headers={"Retry-After": str(retry_after)},
            )
        return user_id
    with request.app.state.chat_rate_lock:
        now = monotonic()
        cutoff = now - _CHAT_WINDOW_SECONDS
        requests = request.app.state.chat_request_times
        # Retire idle users so their counters do not accumulate for the process lifetime.
        for idle_user in [key for key, times in requests.items() if times[-1] <= cutoff]:
            del requests[idle_user]
        times = requests.setdefault(user_id, deque())
        while times and times[0] <= cutoff:
            times.popleft()
        if len(times) >= _CHAT_REQUEST_LIMIT:
            retry_after = max(1, ceil(times[0] + _CHAT_WINDOW_SECONDS - now))
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "rate_limit_exceeded",
                    "message": "Chat is limited to 10 requests per 60 seconds.",
                },
                headers={"Retry-After": str(retry_after)},
            )
        times.append(now)
    return user_id
