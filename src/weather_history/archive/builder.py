"""
Сборка компактного архива метеоданных из одной или нескольких операционных БД.

Поддерживаются два режима:
- replace: полная пересборка во временный *.tmp файл с атомарной публикацией;
- append: дозапись новых данных в уже существующий архив без его пересоздания.

Поведение:
- append=False, overwrite=False:
    ошибка, если output уже существует;
- append=False, overwrite=True:
    полная пересборка output через временный файл;
- append=True:
    существующий output открывается и пополняется;
    если output ещё нет, он будет создан.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from .repository import encode_coordinate, encode_day, encode_value

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceBuildStats:
    path: Path
    points_added: int
    days_added: int
    duplicate_days: int
    skipped: bool = False


@dataclass(frozen=True)
class BuildStats:
    output_path: Path
    profile: str
    point_count: int
    day_count: int
    sources: list[SourceBuildStats]


class CompactWeatherDatabaseBuilder:
    """
    Сборка компактной SQLite-БД из набора рабочих weather БД.

    Ожидаемая исходная схема:
    - weather_sources(id, model, ...)
    - weather_grid_points(id, req_lat, req_lon, ...)
    - weather_days(grid_point_id, source_id, obs_date, wind_speed_max, ...)
    """

    def __init__(
        self,
        output_path: str | Path,
        model: str = "era5",
        overwrite: bool = False,
        append: bool = False,
        cache_mib: int = 256,
        page_size: int = 16_384,
    ) -> None:
        self.output_path = Path(output_path)
        self.model = model
        self.overwrite = overwrite
        self.append = append
        self.cache_mib = cache_mib
        self.page_size = page_size

    def build(self, sources: Iterable[Path]) -> BuildStats:
        output = self.output_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        if self.append:
            return self._build_append(output, sources)

        return self._build_replace(output, sources)

    def _build_replace(self, output: Path, sources: Iterable[Path]) -> BuildStats:
        if output.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output archive already exists: {output}. "
                "Use overwrite=True to rebuild it."
            )

        tmp_output = output.with_name(f"{output.name}.tmp")

        if tmp_output.exists():
            tmp_output.unlink()

        try:
            self._init_db(tmp_output)

            source_stats = self._merge_sources_into_target(
                target_path=tmp_output,
                output_path=output,
                temp_path=tmp_output,
                sources=sources,
            )

            with sqlite3.connect(tmp_output) as dst:
                dst.row_factory = sqlite3.Row
                point_count = dst.execute(
                    "SELECT COUNT(*) AS value FROM weather_grid_points"
                ).fetchone()["value"]
                day_count = dst.execute(
                    "SELECT COUNT(*) AS value FROM weather_days"
                ).fetchone()["value"]

            os.replace(tmp_output, output)

            return BuildStats(
                output_path=output,
                profile="compact-int-v1",
                point_count=point_count,
                day_count=day_count,
                sources=source_stats,
            )

        except Exception:
            if tmp_output.exists():
                tmp_output.unlink()
            raise

    def _build_append(self, output: Path, sources: Iterable[Path]) -> BuildStats:
        if not output.exists():
            self._init_db(output)

        source_stats = self._merge_sources_into_target(
            target_path=output,
            output_path=output,
            temp_path=None,
            sources=sources,
        )

        with sqlite3.connect(output) as dst:
            dst.row_factory = sqlite3.Row
            point_count = dst.execute(
                "SELECT COUNT(*) AS value FROM weather_grid_points"
            ).fetchone()["value"]
            day_count = dst.execute(
                "SELECT COUNT(*) AS value FROM weather_days"
            ).fetchone()["value"]

        return BuildStats(
            output_path=output,
            profile="compact-int-v1",
            point_count=point_count,
            day_count=day_count,
            sources=source_stats,
        )

    def _merge_sources_into_target(
        self,
        target_path: Path,
        output_path: Path,
        temp_path: Path | None,
        sources: Iterable[Path],
    ) -> list[SourceBuildStats]:
        source_stats: list[SourceBuildStats] = []

        with sqlite3.connect(target_path) as dst:
            dst.row_factory = sqlite3.Row
            self._configure_connection(dst)

            for source_path in sources:
                src_path = Path(source_path).expanduser().resolve()
                if src_path == output_path:
                    continue
                if temp_path is not None and src_path == temp_path:
                    continue

                stats = self._merge_one_source(dst, src_path)
                source_stats.append(stats)

            dst.commit()

        return source_stats

    def _init_db(self, output: Path) -> None:
        with sqlite3.connect(output) as con:
            self._configure_connection(con)
            con.executescript(
                """
                PRAGMA journal_mode = DELETE;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS weather_grid_points (
                    id INTEGER PRIMARY KEY,
                    lat_i INTEGER NOT NULL,
                    lon_i INTEGER NOT NULL,
                    UNIQUE(lat_i, lon_i)
                ) STRICT;

                CREATE TABLE IF NOT EXISTS weather_days (
                    point_id INTEGER NOT NULL,
                    day INTEGER NOT NULL,
                    wind_speed_max_i INTEGER,
                    wind_speed_mean_i INTEGER,
                    wind_gust_max_i INTEGER,
                    wind_direction_i INTEGER,
                    PRIMARY KEY (point_id, day),
                    FOREIGN KEY (point_id) REFERENCES weather_grid_points(id)
                ) STRICT, WITHOUT ROWID;
                """
            )
            con.commit()

    def _configure_connection(self, con: sqlite3.Connection) -> None:
        con.execute(f"PRAGMA page_size = {self.page_size}")
        con.execute(f"PRAGMA cache_size = {-self.cache_mib * 1024}")
        con.execute("PRAGMA foreign_keys = ON")

    def _merge_one_source(
        self,
        dst: sqlite3.Connection,
        source_path: Path,
    ) -> SourceBuildStats:
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        src = sqlite3.connect(source_path)
        src.row_factory = sqlite3.Row

        try:
            source_ids = self._get_source_ids_for_model(src, self.model)
            if not source_ids:
                return SourceBuildStats(
                    path=source_path,
                    points_added=0,
                    days_added=0,
                    duplicate_days=0,
                    skipped=True,
                )

            point_id_map: dict[int, int] = {}
            points_added = 0
            days_added = 0
            duplicate_days = 0

            src_points = src.execute(
                """
                SELECT id, req_lat, req_lon
                FROM weather_grid_points
                """
            ).fetchall()

            for row in src_points:
                dst_point_id, added = self._get_or_create_point(
                    dst=dst,
                    lat=float(row["req_lat"]),
                    lon=float(row["req_lon"]),
                )
                point_id_map[int(row["id"])] = dst_point_id
                if added:
                    points_added += 1

            placeholders = ",".join("?" for _ in source_ids)
            src_days = src.execute(
                f"""
                SELECT
                    grid_point_id,
                    obs_date,
                    wind_speed_max,
                    wind_speed_mean,
                    wind_gust_max,
                    wind_direction
                FROM weather_days
                WHERE source_id IN ({placeholders})
                ORDER BY grid_point_id, obs_date
                """,
                tuple(source_ids),
            ).fetchall()

            for row in src_days:
                src_point_id = int(row["grid_point_id"])
                dst_point_id = point_id_map[src_point_id]
                day = self._normalize_date(row["obs_date"])

                payload = {
                    "point_id": dst_point_id,
                    "day": encode_day(day),
                    "wind_speed_max_i": self._encode_optional(row["wind_speed_max"]),
                    "wind_speed_mean_i": self._encode_optional(row["wind_speed_mean"]),
                    "wind_gust_max_i": self._encode_optional(row["wind_gust_max"]),
                    "wind_direction_i": self._encode_optional(row["wind_direction"]),
                }

                inserted = self._insert_or_validate_day(dst, payload)
                if inserted:
                    days_added += 1
                else:
                    duplicate_days += 1

            dst.commit()

            return SourceBuildStats(
                path=source_path,
                points_added=points_added,
                days_added=days_added,
                duplicate_days=duplicate_days,
                skipped=False,
            )
        finally:
            src.close()

    def _get_source_ids_for_model(
        self,
        src: sqlite3.Connection,
        model: str,
    ) -> list[int]:
        rows = src.execute(
            """
            SELECT id
            FROM weather_sources
            WHERE model = ?
            """,
            (model,),
        ).fetchall()
        return [int(row["id"]) for row in rows]

    def _get_or_create_point(
        self,
        dst: sqlite3.Connection,
        lat: float,
        lon: float,
    ) -> tuple[int, bool]:
        lat_i = encode_coordinate(lat)
        lon_i = encode_coordinate(lon)

        row = dst.execute(
            """
            SELECT id
            FROM weather_grid_points
            WHERE lat_i = ? AND lon_i = ?
            """,
            (lat_i, lon_i),
        ).fetchone()
        if row is not None:
            return int(row["id"]), False

        cursor = dst.execute(
            """
            INSERT INTO weather_grid_points (lat_i, lon_i)
            VALUES (?, ?)
            """,
            (lat_i, lon_i),
        )
        return int(cursor.lastrowid), True

    def _insert_or_validate_day(
        self,
        dst: sqlite3.Connection,
        payload: dict,
    ) -> bool:
        existing = dst.execute(
            """
            SELECT
                wind_speed_max_i,
                wind_speed_mean_i,
                wind_gust_max_i,
                wind_direction_i
            FROM weather_days
            WHERE point_id = ? AND day = ?
            """,
            (payload["point_id"], payload["day"]),
        ).fetchone()

        if existing is None:
            dst.execute(
                """
                INSERT INTO weather_days (
                    point_id, day,
                    wind_speed_max_i, wind_speed_mean_i,
                    wind_gust_max_i, wind_direction_i
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["point_id"],
                    payload["day"],
                    payload["wind_speed_max_i"],
                    payload["wind_speed_mean_i"],
                    payload["wind_gust_max_i"],
                    payload["wind_direction_i"],
                ),
            )
            return True

        same = (
            existing["wind_speed_max_i"] == payload["wind_speed_max_i"]
            and existing["wind_speed_mean_i"] == payload["wind_speed_mean_i"]
            and existing["wind_gust_max_i"] == payload["wind_gust_max_i"]
            and existing["wind_direction_i"] == payload["wind_direction_i"]
        )
        if same:
            return False

        raise ValueError(
            f"conflict for point_id={payload['point_id']} day={payload['day']}"
        )

    @staticmethod
    def _normalize_date(value) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    @staticmethod
    def _encode_optional(value: float | int | None) -> int | None:
        if value is None:
            return None
        return encode_value(float(value))


def discover_databases(
    sources: Iterable[str | Path],
    pattern: str = "*.db",
    recursive: bool = False,
) -> list[Path]:
    result: list[Path] = []

    for item in sources:
        path = Path(item).expanduser().resolve()
        if path.is_file():
            result.append(path)
            continue

        if path.is_dir():
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            result.extend(sorted(p.resolve() for p in iterator if p.is_file()))
            continue

        raise FileNotFoundError(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in result:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique