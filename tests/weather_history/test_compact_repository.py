from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from src.weather_history.archive.repository import (
    CompactWeatherRepository,
    GridPoint,
    WeatherRecord,
    decode_coordinate,
    decode_day,
    decode_value,
    encode_coordinate,
    encode_day,
    encode_value,
)


def build_test_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE weather_grid_points (
            id INTEGER PRIMARY KEY,
            lat_i INTEGER NOT NULL,
            lon_i INTEGER NOT NULL,
            UNIQUE(lat_i, lon_i)
        ) STRICT;

        CREATE TABLE weather_days (
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

    con.executemany(
        """
        INSERT INTO weather_grid_points (id, lat_i, lon_i)
        VALUES (?, ?, ?)
        """,
        [
            (1, encode_coordinate(44.50), encode_coordinate(37.75)),
            (2, encode_coordinate(44.75), encode_coordinate(38.00)),
        ],
    )

    con.executemany(
        """
        INSERT INTO weather_days (
            point_id, day,
            wind_speed_max_i, wind_speed_mean_i, wind_gust_max_i, wind_direction_i
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                encode_day(date(2010, 8, 15)),
                encode_value(12.34),
                encode_value(8.90),
                encode_value(15.67),
                encode_value(180.0),
            ),
            (
                1,
                encode_day(date(2010, 8, 16)),
                encode_value(10.0),
                encode_value(7.5),
                encode_value(12.0),
                encode_value(200.0),
            ),
            (
                2,
                encode_day(date(2010, 8, 15)),
                encode_value(20.0),
                encode_value(14.0),
                encode_value(25.0),
                encode_value(220.0),
            ),
        ],
    )
    con.commit()
    con.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "weather_compact_test.db"
    build_test_db(path)
    return path


@pytest.fixture()
def repo(db_path: Path):
    with CompactWeatherRepository(db_path) as repository:
        yield repository


def test_encode_decode_day_roundtrip():
    value = date(2010, 8, 15)
    assert decode_day(encode_day(value)) == value


def test_encode_decode_value_roundtrip():
    value = 12.34
    assert decode_value(encode_value(value)) == pytest.approx(value)


def test_encode_decode_coordinate_roundtrip():
    value = 44.625
    assert decode_coordinate(encode_coordinate(value)) == pytest.approx(value, abs=1e-3)


def test_list_grid_points(repo: CompactWeatherRepository):
    points = repo.list_grid_points()
    assert len(points) == 2
    assert isinstance(points[0], GridPoint)
    assert points[0].lat == pytest.approx(44.50)
    assert points[0].lon == pytest.approx(37.75)


def test_get_point_by_id(repo: CompactWeatherRepository):
    point = repo.get_point_by_id(1)
    assert point is not None
    assert point.id == 1
    assert point.lat == pytest.approx(44.50)
    assert point.lon == pytest.approx(37.75)


def test_nearest_grid_point(repo: CompactWeatherRepository):
    point = repo.nearest_grid_point(44.6, 37.8)
    assert point is not None
    assert point.id == 1


def test_containing_grid_cell_alias(repo: CompactWeatherRepository):
    point = repo.containing_grid_cell(44.6, 37.8)
    assert point is not None
    assert point.id == 1


def test_get_record_for_point(repo: CompactWeatherRepository):
    point = repo.get_point_by_id(1)
    record = repo.get_record_for_point(point, date(2010, 8, 15))
    assert record is not None
    assert isinstance(record, WeatherRecord)
    assert record.obs_date == date(2010, 8, 15)
    assert record.wind_speed_max == pytest.approx(12.34)
    assert record.wind_speed_mean == pytest.approx(8.90)
    assert record.wind_gust_max == pytest.approx(15.67)
    assert record.wind_direction == pytest.approx(180.0)


def test_get_record_for_coordinates(repo: CompactWeatherRepository):
    record = repo.get_record(44.6, 37.8, date(2010, 8, 15))
    assert record is not None
    assert record.point.id == 1
    assert record.wind_speed_max == pytest.approx(12.34)


def test_get_record_returns_none_for_missing_date(repo: CompactWeatherRepository):
    record = repo.get_record(44.6, 37.8, date(1999, 1, 1))
    assert record is None


def test_get_nearest_point_record(repo: CompactWeatherRepository):
    point, record = repo.get_nearest_point_record(44.6, 37.8, date(2010, 8, 15))
    assert point is not None
    assert record is not None
    assert point.id == 1
    assert record.point.id == 1


def test_get_timeseries_for_point_object_api(repo: CompactWeatherRepository):
    point = repo.get_point_by_id(1)
    series = repo.get_timeseries_for_point(point)
    assert len(series) == 2
    assert isinstance(series[0], WeatherRecord)
    assert series[0].obs_date == date(2010, 8, 15)
    assert series[1].obs_date == date(2010, 8, 16)


def test_get_timeseries_for_point_coordinate_api(repo: CompactWeatherRepository):
    series = repo.get_timeseries_for_point(
        44.6,
        37.8,
        date(2010, 8, 15),
        date(2010, 8, 16),
    )
    assert len(series) == 2
    assert series[0].point.id == 1


def test_get_timeseries_with_filter(repo: CompactWeatherRepository):
    series = repo.get_timeseries(
        44.6,
        37.8,
        start_date=date(2010, 8, 16),
        end_date=date(2010, 8, 16),
    )
    assert len(series) == 1
    assert series[0].obs_date == date(2010, 8, 16)


def test_info_contains_expected_keys(repo: CompactWeatherRepository):
    info = repo.info()
    assert info["grid_points"] == 2
    assert info["weather_days"] == 3
    assert info["point_count"] == 2
    assert info["day_count"] == 3
    assert info["date_min"] == date(2010, 8, 15)
    assert info["date_max"] == date(2010, 8, 16)


def test_get_dataframe(repo: CompactWeatherRepository):
    pytest.importorskip("pandas")
    df = repo.get_dataframe(
        44.6,
        37.8,
        start_date=date(2010, 8, 15),
        end_date=date(2010, 8, 16),
    )
    assert len(df) == 2
    assert "wind_speed_max" in df.columns
    assert "point_lat" in df.columns
    assert str(df.index.__class__.__name__) == "DatetimeIndex"
    assert df.iloc[0]["wind_speed_max"] == pytest.approx(12.34)


def test_get_dataframe_for_point(repo: CompactWeatherRepository):
    pytest.importorskip("pandas")
    point = repo.get_point_by_id(2)
    df = repo.get_dataframe_for_point(point)
    assert len(df) == 1
    assert df.iloc[0]["wind_direction"] == pytest.approx(220.0)
