from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# db.py живёт в src/coastline/storage/db.py → parents[3] — корень проекта.
# Стандартное расположение базы: db_pypeline/data/db/coastline.db
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _PROJECT_ROOT / "db_pypeline" / "data" / "db" / "coastline.db"

DATABASE_URL: str = os.environ.get(
    "COASTLINE_DATABASE_URL",
    f"sqlite:///{_DEFAULT_DB.as_posix()}",
)

# SQLite не создаёт промежуточные каталоги — делаем это сами.
if DATABASE_URL.startswith("sqlite:///"):
    _db_path = Path(DATABASE_URL.removeprefix("sqlite:///"))
    if _db_path.is_absolute():
        _db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)