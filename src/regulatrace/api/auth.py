"""Minimal password authentication and typed, expiring JWTs; no token revocation."""

import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Literal
from uuid import UUID, uuid4

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import HashingError, InvalidHashError, VerificationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from regulatrace.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from regulatrace.config import get_settings
from regulatrace.db.repository import find_user_by_email, find_user_by_id, insert_user
from regulatrace.tracing import record_user_role

_PASSWORDS = PasswordHasher()
_ACCESS_LIFETIME = timedelta(minutes=15)
_REFRESH_LIFETIME = timedelta(days=7)


class AuthError(Exception):
    """Safe authentication failure for translation at the HTTP boundary."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _jwt_key() -> bytes:
    configured = get_settings().jwt_secret
    try:
        key = bytes.fromhex(configured.get_secret_value()) if configured else b""
    except ValueError:
        key = b""
    # Require the documented 256-bit format and reject obviously repetitive keys.
    if len(key) != 32 or len(set(key)) < 16:
        raise AuthError(503, "auth_unavailable", "Authentication is not configured securely.")
    return key


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """Do password verification work for unknown emails without a usable account."""
    return _PASSWORDS.hash(secrets.token_urlsafe(32))


def _create_token(user_id: UUID, token_type: Literal["access", "refresh"]) -> str:
    now = datetime.now(timezone.utc)
    lifetime = _ACCESS_LIFETIME if token_type == "access" else _REFRESH_LIFETIME
    return jwt.encode(
        {
            "sub": str(user_id),
            "type": token_type,
            "iat": now,
            "exp": now + lifetime,
            "jti": str(uuid4()),
            "iss": "regulatrace",
            "aud": "regulatrace-api",
        },
        _jwt_key(),
        algorithm="HS256",
    )


def _tokens(user_id: UUID, refresh_token: str | None = None) -> TokenResponse:
    return TokenResponse(
        access_token=_create_token(user_id, "access"),
        # Refresh keeps its original expiry; no pretend rotation or revocation.
        refresh_token=refresh_token or _create_token(user_id, "refresh"),
    )


def token_user_id(
    session: Session, token: str, expected_type: Literal["access", "refresh"]
) -> UUID:
    key = _jwt_key()
    try:
        identities = [("regulatrace", "regulatrace-api")]
        settings = get_settings()
        if (
            settings.previous_token_issuer
            and datetime.now(timezone.utc).timestamp() < settings.previous_token_valid_until
        ):
            identities.append(
                (settings.previous_token_issuer, settings.previous_token_issuer + "-api")
            )
        for issuer, audience in identities:
            try:
                claims = jwt.decode(
                    token,
                    key,
                    algorithms=["HS256"],
                    issuer=issuer,
                    audience=audience,
                    options={"require": ["sub", "type", "exp", "iat", "jti", "iss", "aud"]},
                )
                break
            except (jwt.InvalidIssuerError, jwt.InvalidAudienceError):
                continue
        else:
            raise ValueError("Token identity is not recognized")
        if claims["type"] != expected_type:
            raise ValueError("Wrong token type")
        user_id = UUID(claims["sub"])
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise AuthError(401, "invalid_token", "Invalid or expired token.") from None
    try:
        # Finish the lookup transaction before a slow RAG request uses this session.
        with session.begin():
            user = find_user_by_id(session, user_id)
            if user is None:
                raise AuthError(401, "invalid_token", "Invalid or expired token.")
            record_user_role(user.role)
    except SQLAlchemyError:
        raise AuthError(
            503, "auth_unavailable", "Authentication is temporarily unavailable."
        ) from None
    return user_id


def register_user(session: Session, payload: RegisterRequest) -> UserResponse:
    _jwt_key()
    try:
        user = insert_user(
            session, str(payload.email), _PASSWORDS.hash(payload.password.get_secret_value())
        )
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint == "uq_users_email_lower":
            raise AuthError(409, "email_unavailable", "This email cannot be registered.") from None
        raise AuthError(
            503, "auth_unavailable", "Registration is temporarily unavailable."
        ) from None
    except (SQLAlchemyError, HashingError):
        raise AuthError(
            503, "auth_unavailable", "Registration is temporarily unavailable."
        ) from None
    return UserResponse.model_validate(user, from_attributes=True)


def login_user(session: Session, payload: LoginRequest) -> TokenResponse:
    _jwt_key()
    try:
        with session.begin():
            user = find_user_by_email(session, str(payload.email))
            user_id = user.id if user is not None else None
            password_hash = user.password_hash if user is not None else _dummy_hash()
    except (SQLAlchemyError, HashingError):
        raise AuthError(
            503, "auth_unavailable", "Authentication is temporarily unavailable."
        ) from None
    try:
        _PASSWORDS.verify(password_hash, payload.password.get_secret_value())
    except (VerificationError, InvalidHashError):
        raise AuthError(401, "invalid_credentials", "Incorrect email or password.") from None
    if user_id is None:
        raise AuthError(401, "invalid_credentials", "Incorrect email or password.")
    return _tokens(user_id)


def refresh_tokens(session: Session, token: str) -> TokenResponse:
    return _tokens(token_user_id(session, token, "refresh"), refresh_token=token)
