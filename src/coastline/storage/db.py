# src/coastline/storage/db.py
"""
Подключение к БД береговых точек через SQLAlchemy.
По умолчанию — SQLite, для PostgreSQL задать переменную
окружения COASTLINE_DATABASE_URL.

Примеры:
    sqlite (по умолчанию):
        COASTLINE_DATABASE_URL не задана → sqlite:///data/db/coastline.db

    PostgreSQL:
        export COASTLINE_DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/breachthebeach"
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def build_database_url(db_path: Path | None = None) -> str:
    url = os.getenv("COASTLINE_DATABASE_URL")
    if url:
        return url
    resolved = db_path or Path("data/db/coastline.db")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved.as_posix()}"


def make_engine(db_path: Path | None = None) -> Engine:
    url = build_database_url(db_path)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=False, future=True, connect_args=connect_args)


def make_session_factory(db_path: Path | None = None):
    """
    Создаёт движок для конкретного файла БД, инициализирует схему
    и возвращает фабрику сессий.
    Используется когда путь к БД известен только в момент вызова
    (например, в SQLitePointExporter).
    """
    eng = make_engine(db_path)
    from src.coastline.storage import models  # noqa: F401  (регистрация моделей)
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng, expire_on_commit=False, future=True)


# ---------------------------------------------------------------------------
# Глобальный движок и сессия — для кода, работающего с дефолтной БД
# ---------------------------------------------------------------------------

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
    """Создаёт таблицы в дефолтной БД, если их ещё нет."""
    from src.coastline.storage import models  # noqa: F401
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Контекстный менеджер для дефолтной БД."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
