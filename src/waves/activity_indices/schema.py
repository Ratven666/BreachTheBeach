from __future__ import annotations

from .constants import MAX_DAY_OFFSET

SCHEMA_SQL = f"""
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE points (
    point_id INTEGER PRIMARY KEY,
    lon REAL NOT NULL,
    lat REAL NOT NULL,
    normal_azimuth_deg INTEGER NOT NULL CHECK (normal_azimuth_deg BETWEEN 0 AND 359)
) STRICT;

-- start_day / end_day — целые дни от EPOCH, границы включительные.
CREATE TABLE periods (
    period_id INTEGER PRIMARY KEY,
    start_day INTEGER NOT NULL CHECK (start_day BETWEEN 0 AND {MAX_DAY_OFFSET}),
    end_day INTEGER NOT NULL CHECK (end_day BETWEEN 0 AND {MAX_DAY_OFFSET}),
    n_days INTEGER NOT NULL CHECK (n_days >= 1),
    is_complete INTEGER NOT NULL CHECK (is_complete IN (0, 1)),
    mid_year REAL NOT NULL,
    CHECK (end_day >= start_day)
) STRICT;

CREATE TABLE point_period_indices (
    point_id INTEGER NOT NULL,
    period_id INTEGER NOT NULL,
    n_active_days INTEGER NOT NULL CHECK (n_active_days >= 0),

    normal_mean_wm REAL NOT NULL CHECK (normal_mean_wm >= 0.0),
    alongshore_net_mean_wm REAL NOT NULL,
    alongshore_abs_mean_wm REAL NOT NULL CHECK (alongshore_abs_mean_wm >= 0.0),
    alongshore_pos_mean_wm REAL NOT NULL CHECK (alongshore_pos_mean_wm >= 0.0),
    alongshore_neg_mean_wm REAL NOT NULL CHECK (alongshore_neg_mean_wm >= 0.0),

    d_normal_wm REAL,
    d_normal_pct REAL,
    d_alongshore_abs_wm REAL,
    d_alongshore_abs_pct REAL,
    d_alongshore_net_wm REAL,

    PRIMARY KEY (point_id, period_id),
    FOREIGN KEY (point_id) REFERENCES points(point_id) ON DELETE CASCADE,
    FOREIGN KEY (period_id) REFERENCES periods(period_id) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE TABLE point_trends (
    point_id INTEGER PRIMARY KEY,
    n_periods INTEGER NOT NULL CHECK (n_periods >= 0),

    normal_mean_wm REAL,
    normal_cv REAL,
    normal_slope_wm_per_year REAL,
    normal_r2 REAL,
    normal_first_last_pct REAL,

    alongshore_abs_mean_wm REAL,
    alongshore_abs_cv REAL,
    alongshore_abs_slope_wm_per_year REAL,
    alongshore_abs_r2 REAL,
    alongshore_abs_first_last_pct REAL,

    alongshore_net_mean_wm REAL,
    alongshore_net_slope_wm_per_year REAL,
    dominant_alongshore_sign INTEGER CHECK (
        dominant_alongshore_sign IS NULL OR dominant_alongshore_sign IN (-1, 0, 1)
    ),

    FOREIGN KEY (point_id) REFERENCES points(point_id) ON DELETE CASCADE
) STRICT;

CREATE INDEX idx_indices_period ON point_period_indices(period_id);
"""
