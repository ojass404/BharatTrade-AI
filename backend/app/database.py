from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
    """Base class inherited by all SQLAlchemy database models."""

    pass


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,
)


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Provide one database session for a FastAPI request."""

    database_session = SessionLocal()

    try:
        yield database_session
    finally:
        database_session.close()