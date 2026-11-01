from __future__ import annotations

from datetime import date

EPOCH = date(1940, 1, 1)
EPOCH_TEXT = EPOCH.isoformat()
VALUE_SCALE = 100
COORD_SCALE = 1_000
MAX_DAY_OFFSET = 65_535
SCHEMA_VERSION = 1

SCHEMA_SQL = f"""
CREATE TABLE weather_grid_points (
    id INTEGER PRIMARY KEY,
    lat_i INTEGER NOT NULL CHECK (lat_i BETWEEN {-90 * COORD_SCALE} AND {90 * COORD_SCALE}),
    lon_i INTEGER NOT NULL CHECK (lon_i BETWEEN {-180 * COORD_SCALE} AND {180 * COORD_SCALE}),
    UNIQUE (lat_i, lon_i)
) STRICT;

CREATE TABLE weather_days (
    point_id INTEGER NOT NULL,
    day INTEGER NOT NULL CHECK (day BETWEEN 0 AND {MAX_DAY_OFFSET}),
    wind_speed_max_i INTEGER CHECK (wind_speed_max_i IS NULL OR wind_speed_max_i >= 0),
    wind_speed_mean_i INTEGER CHECK (wind_speed_mean_i IS NULL OR wind_speed_mean_i >= 0),
    wind_gust_max_i INTEGER CHECK (wind_gust_max_i IS NULL OR wind_gust_max_i >= 0),
    wind_direction_i INTEGER CHECK (
        wind_direction_i IS NULL OR
        wind_direction_i BETWEEN 0 AND {360 * VALUE_SCALE}
    ),
    PRIMARY KEY (point_id, day),
    FOREIGN KEY (point_id) REFERENCES weather_grid_points(id) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;
"""
