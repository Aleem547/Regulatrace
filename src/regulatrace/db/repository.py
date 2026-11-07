"""Database connectivity, user lookup and atomic exchange persistence."""

from uuid import UUID

from qdrant_client import QdrantClient
from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from regulatrace.db.models import ChatSession, Message, QueryLog, User


def check_postgres(session: Session) -> None:
    session.execute(text("SELECT 1"))


def save_query_log(session_factory, values: dict) -> None:
    """Use a separate short transaction, after chat messages have committed."""
    with session_factory() as session, session.begin():
        session.add(QueryLog(**values))


def check_qdrant(client: QdrantClient) -> None:
    client.get_collections()


def find_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(func.lower(User.email) == email))


def find_user_by_id(session: Session, user_id: UUID) -> User | None:
    return session.get(User, user_id)


def insert_user(session: Session, email: str, password_hash: str) -> User:
    with session.begin():
        user = User(email=email, password_hash=password_hash, role="user")
        session.add(user)
        session.flush()
        # Preserve returned scalar fields without a post-commit database read.
        session.expunge(user)
    return user


def find_owned_session(session: Session, session_id: UUID, user_id: UUID) -> ChatSession | None:
    return session.scalar(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
    )


def list_owned_sessions(session: Session, user_id: UUID) -> list[ChatSession]:
    return list(
        session.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
            .limit(100)
        )
    )


def recent_session_messages(
    session: Session, session_id: UUID, user_id: UUID, *, limit: int
) -> list[Message]:
    """Fetch a bounded recent tail, then return it in chronological order.

    Pair members share PostgreSQL's transaction timestamp; role breaks that tie
    so a user's question precedes its assistant answer. IDs make remaining ties stable.
    """
    messages = list(
        session.scalars(
            select(Message)
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .order_by(
                Message.created_at.desc(),
                case((Message.role == "assistant", 1), else_=0).desc(),
                Message.id.desc(),
            )
            .limit(limit)
        )
    )
    return messages[::-1]


def save_chat_exchange(
    session: Session,
    session_id: UUID,
    question: str,
    answer: str,
    citations: list[dict],
    *,
    user_id: UUID,
    new_session: bool = True,
) -> None:
    """Commit both messages together, rechecking ownership when appending."""
    with session.begin():
        if new_session:
            session.add(ChatSession(id=session_id, user_id=user_id, title=question.strip()[:120]))
            session.flush()
        else:
            # Recheck after generation and serialize appends without holding a lock during RAG.
            owned = session.scalar(
                select(ChatSession)
                .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .with_for_update()
            )
            if owned is None:
                raise LookupError("Session not found")
        session.add_all(
            [
                Message(session_id=session_id, role="user", content=question, citations=[]),
                Message(
                    session_id=session_id, role="assistant", content=answer, citations=citations
                ),
            ]
        )
