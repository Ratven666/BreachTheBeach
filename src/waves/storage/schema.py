from __future__ import annotations

from src.weather_history.archive.schema import MAX_DAY_OFFSET

SCHEMA_VERSION = 1

SCHEMA_SQL = f"""
CREATE TABLE wave_points (
    id INTEGER PRIMARY KEY,
    lon REAL NOT NULL CHECK (lon BETWEEN -180.0 AND 180.0),
    lat REAL NOT NULL CHECK (lat BETWEEN -90.0 AND 90.0),
    normal_x REAL NOT NULL,
    normal_y REAL NOT NULL,
    normal_azimuth_deg REAL NOT NULL CHECK (
        normal_azimuth_deg >= 0.0 AND normal_azimuth_deg < 360.0
    )
) STRICT;

CREATE TABLE wave_activity (
    point_id INTEGER NOT NULL,
    day INTEGER NOT NULL CHECK (day BETWEEN 0 AND {MAX_DAY_OFFSET}),
    wind_azimuth_deg INTEGER NOT NULL CHECK (
        wind_azimuth_deg BETWEEN 0 AND 359
    ),
    wave_power_wm INTEGER NOT NULL CHECK (wave_power_wm >= 0),
    PRIMARY KEY (point_id, day),
    FOREIGN KEY (point_id) REFERENCES wave_points(id) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE TABLE wave_activity_summary (
    point_id INTEGER PRIMARY KEY,
    n_days INTEGER NOT NULL CHECK (n_days >= 0),
    n_active_days INTEGER NOT NULL CHECK (n_active_days >= 0),
    mean_power_wm REAL,
    median_power_wm REAL,
    std_power_wm REAL,
    p75_power_wm REAL,
    p90_power_wm REAL,
    p95_power_wm REAL,
    p99_power_wm REAL,
    max_power_wm INTEGER,
    n_storm_days_p90 INTEGER NOT NULL CHECK (n_storm_days_p90 >= 0),
    total_energy_mjm REAL NOT NULL CHECK (total_energy_mjm >= 0.0),
    FOREIGN KEY (point_id) REFERENCES wave_points(id) ON DELETE CASCADE
) STRICT;
"""
