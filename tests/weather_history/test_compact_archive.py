from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.weather_history.archive import (
    CompactWeatherDatabaseBuilder,
    CompactWeatherRepository,
)


def make_source(path: Path, speed: float = 10.0) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE weather_sources (
            id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            ws_unit TEXT,
            wd_unit TEXT
        );

        CREATE TABLE weather_grid_points (
            id INTEGER PRIMARY KEY,
            req_lat REAL NOT NULL,
            req_lon REAL NOT NULL,
            ring_y INTEGER NOT NULL DEFAULT 0,
            ring_x INTEGER NOT NULL DEFAULT 0,
            cell_id TEXT,
            resolved_lat REAL,
            resolved_lon REAL,
            elevation_m REAL
        );

        CREATE TABLE weather_days (
            id INTEGER PRIMARY KEY,
            grid_point_id INTEGER NOT NULL,
            source_id INTEGER NOT NULL,
            obs_date DATE NOT NULL,
            wind_speed_max REAL,
            wind_speed_mean REAL,
            wind_gust_max REAL,
            wind_direction REAL
        );
        """
    )

    con.execute(
        """
        INSERT INTO weather_sources (id, model, ws_unit, wd_unit)
        VALUES (1, 'era5', 'km/h', '°')
        """
    )

    con.execute(
        """
        INSERT INTO weather_grid_points (
            id, req_lat, req_lon, ring_y, ring_x, cell_id, resolved_lat, resolved_lon, elevation_m
        )
        VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?)
        """,
        (
            1,
            44.125,
            37.875,
            "37.875_44.125",
            44.125,
            37.875,
            5.0,
        ),
    )

    con.execute(
        """
        INSERT INTO weather_days (
            id,
            grid_point_id,
            source_id,
            obs_date,
            wind_speed_max,
            wind_speed_mean,
            wind_gust_max,
            wind_direction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            1,
            1,
            "1940-01-02",
            speed,
            8.0,
            12.0,
            180.0,
        ),
    )

    con.commit()
    con.close()


class CompactArchiveTest(unittest.TestCase):
    def test_build_deduplicates_and_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.db"
            source_b = root / "b.db"
            output = root / "compact.db"

            make_source(source_a)
            make_source(source_b)

            stats = CompactWeatherDatabaseBuilder(output).build(
                [source_a, source_b]
            )
            self.assertEqual(stats.point_count, 1)
            self.assertEqual(stats.day_count, 1)
            self.assertEqual(stats.sources[1].duplicate_days, 1)

            with CompactWeatherRepository(output) as repository:
                rows = repository.get_timeseries_for_point(
                    44.125, 37.875, "1940-01-01", "1940-01-02"
                )

            self.assertEqual(len(rows), 1)

            row = rows[0]
            self.assertEqual(row.obs_date.isoformat(), "1940-01-02")
            self.assertEqual(row.wind_speed_max, 10.0)
            self.assertEqual(row.wind_speed_mean, 8.0)
            self.assertEqual(row.wind_gust_max, 12.0)
            self.assertEqual(row.wind_direction, 180.0)

            self.assertEqual(row.point.lat, 44.125)
            self.assertEqual(row.point.lon, 37.875)

    def test_conflicting_duplicate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.db"
            source_b = root / "b.db"
            output = root / "compact.db"

            make_source(source_a, speed=10.0)
            make_source(source_b, speed=11.0)

            with self.assertRaisesRegex(ValueError, "conflict"):
                CompactWeatherDatabaseBuilder(output).build(
                    [source_a, source_b]
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
