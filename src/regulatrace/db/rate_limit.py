"""A shared rolling chat limit for serverless replicas, backed by PostgreSQL."""

from datetime import datetime, timedelta, timezone
from math import ceil

from sqlalchemy import delete, func, select, text

from regulatrace.db.models import ChatRequestAttempt


def admit_chat_attempt(session, user_id, *, now=None, limit=10, seconds=60) -> int:
    """Return zero on admission or a Retry-After delay, using a transaction-scoped lock."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=seconds)
    lock_key = int.from_bytes(user_id.bytes[-8:], "big", signed=True)
    with session.begin():
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
        session.execute(delete(ChatRequestAttempt).where(ChatRequestAttempt.created_at <= cutoff))
        count, first = session.execute(
            select(func.count(), func.min(ChatRequestAttempt.created_at)).where(
                ChatRequestAttempt.user_id == user_id,
                ChatRequestAttempt.created_at > cutoff,
            )
        ).one()
        if count >= limit:
            return max(1, ceil((first + timedelta(seconds=seconds) - now).total_seconds()))
        session.add(ChatRequestAttempt(user_id=user_id, created_at=now))
    return 0
