from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import SCHEMA_SQL, SCHEMA_VERSION


def round_half_up(value: float) -> int:
    """Округляет до ближайшего целого, половины — вверх (2.5 -> 3).

    Стандартный round() в Python использует банковское округление
    (2.5 -> 2), что для физических величин нежелательно.
    """
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"Cannot round non-finite value: {value}")

    return int(math.floor(value + 0.5))


def round_azimuth_deg(azimuth_deg: float) -> int:
    """Округляет азимут до целого градуса в диапазоне [0, 359].

    Нормализация выполняется после округления, чтобы 359.6 -> 0, а не 360.
    """
    return round_half_up(azimuth_deg) % 360


def round_cwef_wm(cwef_wm: float) -> int:
    """Округляет суточный CWEF до целого, Вт/м."""
    value = float(cwef_wm)

    if value < 0.0:
        raise ValueError(f"Negative CWEF value: {value}")

    return round_half_up(value)


@dataclass(frozen=True, slots=True)
class WavePointRow:
    id: int
    lon: float
    lat: float
    normal_azimuth_deg: int


@dataclass(frozen=True, slots=True)
class WaveActivityRow:
    point_id: int
    day: int
    wind_azimuth_deg: int
    cwef_wm: int


@dataclass(frozen=True, slots=True)
class WaveActivitySummaryRow:
    point_id: int
    n_days: int
    n_active_days: int
    mean_cwef_wm: float | None
    median_cwef_wm: float | None
    std_cwef_wm: float | None
    p75_cwef_wm: float | None
    p90_cwef_wm: float | None
    p95_cwef_wm: float | None
    p99_cwef_wm: float | None
    max_cwef_wm: float | None
    n_storm_days_p90: int
    total_energy_mjm: float

    @classmethod
    def from_powers(
        cls,
        point_id: int,
        powers_wm: Iterable[float],
    ) -> "WaveActivitySummaryRow":
        """Сводная статистика по исходным (неокруглённым) значениям CWEF."""
        values = np.asarray(list(powers_wm), dtype=np.float64)

        if values.size == 0:
            return cls(
                point_id=int(point_id),
                n_days=0,
                n_active_days=0,
                mean_cwef_wm=None,
                median_cwef_wm=None,
                std_cwef_wm=None,
                p75_cwef_wm=None,
                p90_cwef_wm=None,
                p95_cwef_wm=None,
                p99_cwef_wm=None,
                max_cwef_wm=None,
                n_storm_days_p90=0,
                total_energy_mjm=0.0,
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite CWEF values for point_id={point_id}"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"Negative CWEF values for point_id={point_id}"
            )

        p90 = float(np.quantile(values, 0.90))

        return cls(
            point_id=int(point_id),
            n_days=int(values.size),
            n_active_days=int(np.count_nonzero(values > 0.0)),
            mean_cwef_wm=round(float(np.mean(values)), 6),
            median_cwef_wm=round(float(np.median(values)), 6),
            std_cwef_wm=(
                round(float(np.std(values, ddof=1)), 6)
                if values.size > 1
                else None
            ),
            p75_cwef_wm=round(float(np.quantile(values, 0.75)), 6),
            p90_cwef_wm=round(p90, 6),
            p95_cwef_wm=round(float(np.quantile(values, 0.95)), 6),
            p99_cwef_wm=round(float(np.quantile(values, 0.99)), 6),
            max_cwef_wm=round(float(np.max(values)), 6),
            n_storm_days_p90=int(np.count_nonzero(values >= p90)),
            total_energy_mjm=round(
                float(np.sum(values) * 86_400.0 / 1e6),
                6,
            ),
        )


@dataclass(frozen=True, slots=True)
class WaveExposureIndexRow:
    point_id: int
    mean_cwef_wm: float | None
    e_storm_mjm: float | None
    storm_threshold_wm: float | None
    storm_percentile: float | None
    k_dir: float | None
    cv: float | None
    n_days: int
    n_storm_days: int
    top3_sectors: str | None
    r1: int | None
    r2: int | None
    r3: int | None
    r4: int | None
    wer: float | None


class WaveActivityRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._con = sqlite3.connect(self.path)
        self._con.execute("PRAGMA foreign_keys = ON")
        self._con.execute("PRAGMA synchronous = NORMAL")
        self._con.execute("PRAGMA cache_size = -65536")
        self._con.execute("PRAGMA temp_store = MEMORY")

    def __enter__(self) -> "WaveActivityRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is None:
                self._con.commit()
            else:
                self._con.rollback()
        finally:
            self.close()

    def close(self) -> None:
        self._con.close()

    def initialize(self) -> None:
        self._con.executescript(SCHEMA_SQL)
        self._con.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
        self._con.commit()

    def assert_schema_version(self) -> None:
        version = int(
            self._con.execute("PRAGMA user_version").fetchone()[0]
        )

        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported wave database schema version: "
                f"{version}; expected {SCHEMA_VERSION}. "
                "Re-run db_pypeline/5_1_wave_activity_to_db.py."
            )

    def add_point(self, point: WavePointRow) -> None:
        self._con.execute(
            """
            INSERT INTO wave_points (
                id,
                lon,
                lat,
                normal_azimuth_deg
            ) VALUES (?, ?, ?, ?)
            """,
            (
                int(point.id),
                float(point.lon),
                float(point.lat),
                round_azimuth_deg(point.normal_azimuth_deg),
            ),
        )

    def add_activity(
        self,
        rows: Iterable[WaveActivityRow],
    ) -> int:
        """Записывает суточные CWEF, округлённые до целых.

        Дни, у которых округлённый CWEF равен 0, не сохраняются
        (таблица разреженная, CHECK cwef_wm >= 1).
        """
        payload: list[tuple[int, int, int, int]] = []

        for row in rows:
            cwef_int = round_cwef_wm(row.cwef_wm)

            if cwef_int < 1:
                continue

            payload.append(
                (
                    int(row.point_id),
                    int(row.day),
                    int(row.wind_azimuth_deg) % 360,
                    cwef_int,
                )
            )

        if not payload:
            return 0

        self._con.executemany(
            """
            INSERT INTO wave_activity (
                point_id,
                day,
                wind_azimuth_deg,
                cwef_wm
            ) VALUES (?, ?, ?, ?)
            """,
            payload,
        )

        return len(payload)

    def add_summary(
        self,
        row: WaveActivitySummaryRow,
    ) -> None:
        self._con.execute(
            """
            INSERT INTO wave_activity_summary (
                point_id,
                n_days,
                n_active_days,
                mean_cwef_wm,
                median_cwef_wm,
                std_cwef_wm,
                p75_cwef_wm,
                p90_cwef_wm,
                p95_cwef_wm,
                p99_cwef_wm,
                max_cwef_wm,
                n_storm_days_p90,
                total_energy_mjm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row.point_id),
                int(row.n_days),
                int(row.n_active_days),
                row.mean_cwef_wm,
                row.median_cwef_wm,
                row.std_cwef_wm,
                row.p75_cwef_wm,
                row.p90_cwef_wm,
                row.p95_cwef_wm,
                row.p99_cwef_wm,
                row.max_cwef_wm,
                int(row.n_storm_days_p90),
                float(row.total_energy_mjm),
            ),
        )

    def read_summaries(
        self,
    ) -> list[WaveActivitySummaryRow]:
        rows = self._con.execute(
            """
            SELECT
                point_id,
                n_days,
                n_active_days,
                mean_cwef_wm,
                median_cwef_wm,
                std_cwef_wm,
                p75_cwef_wm,
                p90_cwef_wm,
                p95_cwef_wm,
                p99_cwef_wm,
                max_cwef_wm,
                n_storm_days_p90,
                total_energy_mjm
            FROM wave_activity_summary
            ORDER BY point_id
            """
        ).fetchall()

        return [
            WaveActivitySummaryRow(
                point_id=int(row[0]),
                n_days=int(row[1]),
                n_active_days=int(row[2]),
                mean_cwef_wm=row[3],
                median_cwef_wm=row[4],
                std_cwef_wm=row[5],
                p75_cwef_wm=row[6],
                p90_cwef_wm=row[7],
                p95_cwef_wm=row[8],
                p99_cwef_wm=row[9],
                max_cwef_wm=row[10],
                n_storm_days_p90=int(row[11]),
                total_energy_mjm=float(row[12]),
            )
            for row in rows
        ]

    def read_activity_by_point(
        self,
    ) -> dict[int, list[WaveActivityRow]]:
        rows = self._con.execute(
            """
            SELECT
                point_id,
                day,
                wind_azimuth_deg,
                cwef_wm
            FROM wave_activity
            ORDER BY point_id, day
            """
        ).fetchall()

        result: dict[int, list[WaveActivityRow]] = defaultdict(list)

        for row in rows:
            item = WaveActivityRow(
                point_id=int(row[0]),
                day=int(row[1]),
                wind_azimuth_deg=int(row[2]),
                cwef_wm=int(row[3]),
            )
            result[item.point_id].append(item)

        return dict(result)

    def replace_exposure_indices(
        self,
        rows: Iterable[WaveExposureIndexRow],
    ) -> int:
        payload = [
            (
                int(row.point_id),
                row.mean_cwef_wm,
                row.e_storm_mjm,
                row.storm_threshold_wm,
                row.storm_percentile,
                row.k_dir,
                row.cv,
                int(row.n_days),
                int(row.n_storm_days),
                row.top3_sectors,
                row.r1,
                row.r2,
                row.r3,
                row.r4,
                row.wer,
            )
            for row in rows
        ]

        self._con.execute("DELETE FROM wave_exposure_index")

        if payload:
            self._con.executemany(
                """
                INSERT INTO wave_exposure_index (
                    point_id,
                    mean_cwef_wm,
                    e_storm_mjm,
                    storm_threshold_wm,
                    storm_percentile,
                    k_dir,
                    cv,
                    n_days,
                    n_storm_days,
                    top3_sectors,
                    r1,
                    r2,
                    r3,
                    r4,
                    wer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

        return len(payload)

    def commit(self) -> None:
        self._con.commit()

    def optimize(self) -> None:
        self._con.execute("PRAGMA optimize")
        self._con.commit()

    def read_exposure_indices(
        self,
    ) -> list[tuple[WavePointRow, WaveExposureIndexRow]]:
        """Читает wave_points JOIN wave_exposure_index.

        Возвращает список пар (точка, индекс) в порядке point_id.
        Точки без записи в wave_exposure_index не включаются.
        """
        rows = self._con.execute(
            """
            SELECT
                p.id,
                p.lon,
                p.lat,
                p.normal_azimuth_deg,
                i.mean_cwef_wm,
                i.e_storm_mjm,
                i.storm_threshold_wm,
                i.storm_percentile,
                i.k_dir,
                i.cv,
                i.n_days,
                i.n_storm_days,
                i.top3_sectors,
                i.r1,
                i.r2,
                i.r3,
                i.r4,
                i.wer
            FROM wave_points AS p
            INNER JOIN wave_exposure_index AS i
                ON i.point_id = p.id
            ORDER BY p.id
            """
        ).fetchall()

        result: list[tuple[WavePointRow, WaveExposureIndexRow]] = []

        for row in rows:
            point = WavePointRow(
                id=int(row[0]),
                lon=float(row[1]),
                lat=float(row[2]),
                normal_azimuth_deg=int(row[3]),
            )
            index = WaveExposureIndexRow(
                point_id=int(row[0]),
                mean_cwef_wm=row[4],
                e_storm_mjm=row[5],
                storm_threshold_wm=row[6],
                storm_percentile=row[7],
                k_dir=row[8],
                cv=row[9],
                n_days=int(row[10]),
                n_storm_days=int(row[11]),
                top3_sectors=row[12],
                r1=row[13],
                r2=row[14],
                r3=row[15],
                r4=row[16],
                wer=row[17],
            )
            result.append((point, index))

        return result
