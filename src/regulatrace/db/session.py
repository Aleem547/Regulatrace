"""Database session and engine management for Regulatrace."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from regulatrace.config import get_settings


class Base(DeclarativeBase):
    """Base declarative class for all Regulatrace database models."""

    pass


def normalize_database_url(value: str | URL) -> URL:
    """Keep the installed psycopg2 driver for generic PostgreSQL connection URLs."""
    parsed = make_url(value)
    if parsed.drivername in {"postgres", "postgresql"}:
        return parsed.set(drivername="postgresql+psycopg2")
    return parsed


def get_engine(url: str | None = None, **engine_options):
    """Create an engine from configuration, with optional SQLAlchemy connection options."""
    settings = get_settings()
    db_url = url or settings.database_url
    if settings.database_null_pool:
        engine_options["poolclass"] = NullPool
        for key in ("pool_timeout", "pool_size", "max_overflow"):
            engine_options.pop(key, None)
    return create_engine(normalize_database_url(db_url), **engine_options)


def get_session_factory(engine=None) -> sessionmaker[Session]:
    """Return a sessionmaker factory bound to the provided or default engine."""
    eng = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


def get_db(url: str | None = None) -> Generator[Session, None, None]:
    """Context generator yielding a transactional database session."""
    engine = get_engine(url)
    session_factory = get_session_factory(engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
