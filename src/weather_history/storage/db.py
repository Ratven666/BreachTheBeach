"""
src/weather_history/storage/db.py

Подключение к БД через SQLAlchemy.
По умолчанию — SQLite-файл, для перехода на PostgreSQL достаточно
задать переменную окружения DATABASE_URL, код менять не нужно.

Примеры:
    sqlite (по умолчанию):
        DATABASE_URL не задана -> sqlite:///data/db/weather.db

    PostgreSQL:
        export DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/breachthebeach"
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    db_path = Path("data/db/weather.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


DATABASE_URL = build_database_url()

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Создаёт таблицы, если их ещё нет. В проде лучше заменить на Alembic-миграции."""
    from src.weather_history.storage import models  # noqa: F401  (регистрация моделей)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
