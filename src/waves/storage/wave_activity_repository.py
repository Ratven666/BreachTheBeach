from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import SCHEMA_SQL, SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class WavePointRow:
    id: int
    lon: float
    lat: float
    normal_x: float
    normal_y: float
    normal_azimuth_deg: float


@dataclass(frozen=True, slots=True)
class WaveActivityRow:
    point_id: int
    day: int
    wind_azimuth_deg: int
    wave_power_wm: int


@dataclass(frozen=True, slots=True)
class WaveActivitySummaryRow:
    point_id: int
    n_days: int
    n_active_days: int
    mean_power_wm: float | None
    median_power_wm: float | None
    std_power_wm: float | None
    p75_power_wm: float | None
    p90_power_wm: float | None
    p95_power_wm: float | None
    p99_power_wm: float | None
    max_power_wm: int | None
    n_storm_days_p90: int
    total_energy_mjm: float

    @classmethod
    def from_powers(
        cls,
        point_id: int,
        powers_wm: Iterable[int],
    ) -> "WaveActivitySummaryRow":
        values = np.asarray(list(powers_wm), dtype=np.float64)
        if values.size == 0:
            return cls(
                point_id=point_id,
                n_days=0,
                n_active_days=0,
                mean_power_wm=None,
                median_power_wm=None,
                std_power_wm=None,
                p75_power_wm=None,
                p90_power_wm=None,
                p95_power_wm=None,
                p99_power_wm=None,
                max_power_wm=None,
                n_storm_days_p90=0,
                total_energy_mjm=0.0,
            )

        p90 = float(np.quantile(values, 0.90))
        return cls(
            point_id=point_id,
            n_days=int(values.size),
            n_active_days=int(np.count_nonzero(values > 0.0)),
            mean_power_wm=round(float(np.mean(values)), 2),
            median_power_wm=round(float(np.median(values)), 2),
            std_power_wm=(
                round(float(np.std(values, ddof=1)), 2)
                if values.size > 1
                else None
            ),
            p75_power_wm=round(float(np.quantile(values, 0.75)), 2),
            p90_power_wm=round(p90, 2),
            p95_power_wm=round(float(np.quantile(values, 0.95)), 2),
            p99_power_wm=round(float(np.quantile(values, 0.99)), 2),
            max_power_wm=int(np.max(values)),
            n_storm_days_p90=int(np.count_nonzero(values >= p90)),
            total_energy_mjm=round(float(np.sum(values) * 86_400.0 / 1e6), 1),
        )


class WaveActivityRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path)
        self._con.execute("PRAGMA foreign_keys = ON")
        self._con.execute("PRAGMA synchronous = NORMAL")
        self._con.execute("PRAGMA cache_size = -65536")

    def __enter__(self) -> "WaveActivityRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._con.commit()
        else:
            self._con.rollback()
        self.close()

    def close(self) -> None:
        self._con.close()

    def initialize(self) -> None:
        self._con.executescript(SCHEMA_SQL)
        self._con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._con.commit()

    def add_point(self, point: WavePointRow) -> None:
        self._con.execute(
            """
            INSERT INTO wave_points (
                id, lon, lat, normal_x, normal_y, normal_azimuth_deg
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                point.id,
                point.lon,
                point.lat,
                point.normal_x,
                point.normal_y,
                point.normal_azimuth_deg % 360.0,
            ),
        )

    def add_activity(self, rows: Iterable[WaveActivityRow]) -> int:
        payload = [
            (row.point_id, row.day, row.wind_azimuth_deg, row.wave_power_wm)
            for row in rows
        ]
        if not payload:
            return 0
        self._con.executemany(
            """
            INSERT INTO wave_activity (
                point_id, day, wind_azimuth_deg, wave_power_wm
            ) VALUES (?, ?, ?, ?)
            """,
            payload,
        )
        return len(payload)

    def add_summary(self, row: WaveActivitySummaryRow) -> None:
        self._con.execute(
            """
            INSERT INTO wave_activity_summary (
                point_id, n_days, n_active_days,
                mean_power_wm, median_power_wm, std_power_wm,
                p75_power_wm, p90_power_wm, p95_power_wm, p99_power_wm,
                max_power_wm, n_storm_days_p90, total_energy_mjm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.point_id,
                row.n_days,
                row.n_active_days,
                row.mean_power_wm,
                row.median_power_wm,
                row.std_power_wm,
                row.p75_power_wm,
                row.p90_power_wm,
                row.p95_power_wm,
                row.p99_power_wm,
                row.max_power_wm,
                row.n_storm_days_p90,
                row.total_energy_mjm,
            ),
        )

    def commit(self) -> None:
        self._con.commit()

    def optimize(self) -> None:
        self._con.execute("PRAGMA optimize")
        self._con.commit()
