from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .constants import SCHEMA_VERSION, SOURCE_SCHEMA_VERSION, SQLITE_CACHE_KIB
from .schema import SCHEMA_SQL


class SourceActivityReader:
    """Читает wave_activity.db строго в режиме read-only."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Wave activity database not found: {self.path}")
        self._con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        version = int(self._con.execute("PRAGMA user_version").fetchone()[0])
        if version != SOURCE_SCHEMA_VERSION:
            self._con.close()
            raise ValueError(
                f"Unsupported source schema version {version}; expected {SOURCE_SCHEMA_VERSION}"
            )

    def __enter__(self) -> "SourceActivityReader":
        return self

    def __exit__(self, *exc) -> None:
        self._con.close()

    def read_points(self) -> list[tuple[int, float, float, int]]:
        return [
            (int(i), float(lo), float(la), int(az))
            for i, lo, la, az in self._con.execute(
                "SELECT id, lon, lat, normal_azimuth_deg FROM wave_points ORDER BY id"
            )
        ]

    def read_n_days(self) -> dict[int, int]:
        return {
            int(i): int(n)
            for i, n in self._con.execute("SELECT point_id, n_days FROM wave_activity_summary")
        }

    def day_bounds(self) -> tuple[int, int] | None:
        row = self._con.execute("SELECT MIN(day), MAX(day) FROM wave_activity").fetchone()
        return None if row[0] is None else (int(row[0]), int(row[1]))

    def iter_activity(self, point_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._con.execute(
            "SELECT day, wind_azimuth_deg, cwef_wm FROM wave_activity "
            "WHERE point_id = ? ORDER BY day",
            (int(point_id),),
        ).fetchall()
        if not rows:
            e = np.empty(0)
            return e.astype(np.int64), e, e
        a = np.asarray(rows, dtype=np.float64)
        return a[:, 0].astype(np.int64), a[:, 1], a[:, 2]


class IndexRepository:
    """Запись и чтение БД индексов."""

    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(f"Index database not found: {self.path}")
            self._con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(self.path)
            self._con.execute("PRAGMA foreign_keys = ON")
            self._con.execute(f"PRAGMA cache_size = -{SQLITE_CACHE_KIB}")
        self._con.row_factory = sqlite3.Row

    def __enter__(self) -> "IndexRepository":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        try:
            if exc_type is None and self._con.in_transaction:
                self._con.commit()
            elif self._con.in_transaction:
                self._con.rollback()
        finally:
            self._con.close()

    def initialize(self) -> None:
        self._con.executescript(SCHEMA_SQL)
        self._con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def add_meta(self, items: dict[str, object]) -> None:
        self._con.executemany(
            "INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in items.items()]
        )

    def add_many(self, table: str, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        if not rows:
            return
        marks = ",".join("?" * len(rows[0]))
        self._con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)

    def commit(self) -> None:
        self._con.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self._con.execute(sql, params).fetchall()

    def iter_query(self, sql: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
        yield from self._con.execute(sql, params)
