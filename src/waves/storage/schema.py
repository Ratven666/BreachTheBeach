from __future__ import annotations

from src.weather_history.archive.schema import MAX_DAY_OFFSET


SCHEMA_VERSION = 3

SCHEMA_SQL = f"""
CREATE TABLE wave_points (
    id INTEGER PRIMARY KEY,
    lon REAL NOT NULL CHECK (lon BETWEEN -180.0 AND 180.0),
    lat REAL NOT NULL CHECK (lat BETWEEN -90.0 AND 90.0),

    -- Азимут нормали, округлённый до целого градуса: int(round(az)) % 360.
    normal_azimuth_deg INTEGER NOT NULL CHECK (
        normal_azimuth_deg BETWEEN 0 AND 359
    )
) STRICT;

-- Таблица разреженная: в ней хранятся только дни с округлённым CWEF >= 1.
-- Полное число обработанных дней хранится в wave_activity_summary.n_days.
CREATE TABLE wave_activity (
    point_id INTEGER NOT NULL,
    day INTEGER NOT NULL CHECK (
        day BETWEEN 0 AND {MAX_DAY_OFFSET}
    ),
    wind_azimuth_deg INTEGER NOT NULL CHECK (
        wind_azimuth_deg BETWEEN 0 AND 359
    ),

    -- Суточный CWEF, округлённый до целого, Вт/м.
    cwef_wm INTEGER NOT NULL CHECK (
        cwef_wm >= 1
    ),

    PRIMARY KEY (point_id, day),
    FOREIGN KEY (point_id)
        REFERENCES wave_points(id)
        ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE TABLE wave_activity_summary (
    point_id INTEGER PRIMARY KEY,

    -- Все валидные метеорологические дни, включая:
    --   * штиль;
    --   * ветер из сушевого сектора;
    --   * дни с нулевым CWEF.
    n_days INTEGER NOT NULL CHECK (n_days >= 0),

    -- Число дней с CWEF > 0 (до округления).
    n_active_days INTEGER NOT NULL CHECK (
        n_active_days >= 0
        AND n_active_days <= n_days
    ),

    mean_cwef_wm REAL,
    median_cwef_wm REAL,
    std_cwef_wm REAL,
    p75_cwef_wm REAL,
    p90_cwef_wm REAL,
    p95_cwef_wm REAL,
    p99_cwef_wm REAL,
    max_cwef_wm REAL,

    n_storm_days_p90 INTEGER NOT NULL CHECK (
        n_storm_days_p90 >= 0
        AND n_storm_days_p90 <= n_days
    ),

    total_energy_mjm REAL NOT NULL CHECK (
        total_energy_mjm >= 0.0
    ),

    FOREIGN KEY (point_id)
        REFERENCES wave_points(id)
        ON DELETE CASCADE
) STRICT;

CREATE TABLE wave_exposure_index (
    point_id INTEGER PRIMARY KEY,

    mean_cwef_wm REAL,
    e_storm_mjm REAL,
    storm_threshold_wm REAL,
    storm_percentile REAL CHECK (
        storm_percentile IS NULL
        OR (
            storm_percentile > 0.0
            AND storm_percentile < 100.0
        )
    ),

    k_dir REAL CHECK (
        k_dir IS NULL
        OR (
            k_dir >= 0.0
            AND k_dir <= 1.0
        )
    ),

    cv REAL CHECK (
        cv IS NULL
        OR cv >= 0.0
    ),

    n_days INTEGER NOT NULL CHECK (n_days >= 0),
    n_storm_days INTEGER NOT NULL CHECK (
        n_storm_days >= 0
        AND n_storm_days <= n_days
    ),

    top3_sectors TEXT,

    r1 INTEGER CHECK (r1 IS NULL OR r1 BETWEEN 1 AND 5),
    r2 INTEGER CHECK (r2 IS NULL OR r2 BETWEEN 1 AND 5),
    r3 INTEGER CHECK (r3 IS NULL OR r3 BETWEEN 1 AND 5),
    r4 INTEGER CHECK (r4 IS NULL OR r4 BETWEEN 1 AND 5),

    wer REAL CHECK (
        wer IS NULL
        OR (
            wer >= 1.0
            AND wer <= 5.0
        )
    ),

    FOREIGN KEY (point_id)
        REFERENCES wave_points(id)
        ON DELETE CASCADE
) STRICT;

CREATE INDEX idx_wave_activity_day
    ON wave_activity(day);

CREATE INDEX idx_wave_exposure_wer
    ON wave_exposure_index(wer);
"""