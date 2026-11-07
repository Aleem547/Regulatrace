"""Request and response contracts for the Regulatrace API."""

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: dict[str, Literal["ok", "unavailable"]]


class LoginRequest(_RequestModel):
    email: EmailStr
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class RegisterRequest(LoginRequest):
    password: SecretStr = Field(min_length=12, max_length=128)


class RefreshRequest(_RequestModel):
    refresh_token: SecretStr = Field(min_length=1)


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class ChatRequest(_RequestModel):
    question: str = Field(min_length=1, pattern=r"\S")
    session_id: UUID | None = None


class SourceResponse(BaseModel):
    """One source record produced by ``rag.format_evidence``."""

    tag: str
    doc_id: str | None
    chunk_index: int | None
    title: str | None
    source_file: str | None
    page: int | None
    circular_no: str | None
    source_url: str | None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    abstained: bool
    citation_validation: dict[str, Any]
    session_id: UUID


class ChatErrorResponse(ErrorResponse):
    sources: list[SourceResponse] = Field(default_factory=list)


class SessionResponse(BaseModel):
    id: UUID
    created_at: datetime
    title: str | None


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime


class SessionDetailResponse(SessionResponse):
    messages: list[MessageResponse]


class FeedbackRequest(_RequestModel):
    message_id: UUID
    rating: int
    comment: str | None = None


class FeedbackResponse(FeedbackRequest):
    id: UUID
    created_at: datetime


class DocumentResponse(BaseModel):
    id: UUID
    title: str | None
    regulator: str | None
    circular_no: str | None
    issue_date: date | None
    status: str | None = Field(
        description="Verified regulatory status, or null when unverified; not ingestion status."
    )
    access_level: str | None
    chunk_count: int


class IngestRequest(_RequestModel):
    source_file: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Relative source reference for the planned controlled ingest workflow.",
    )


class JobResponse(BaseModel):
    id: UUID
    status: str
