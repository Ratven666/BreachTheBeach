from __future__ import annotations

import sqlite3
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.coastline.storage.models import (
    Base,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
    WindFetchModel,
)
from src.weather_history.archive.repository import InterpolatedWeatherRecord
from src.waves.services.wave_activity_db_service import WaveActivityDatabaseBuilder
from src.waves.storage import WaveActivityRepository


class FakeWeatherRepository:
    def get_interpolated_timeseries(self, lat, lon, start_date=None, end_date=None):
        assert (lat, lon) == (60.0, 30.0)
        return [
            InterpolatedWeatherRecord(
                obs_date=date(1940, 1, 2),
                query_lat=lat,
                query_lon=lon,
                wind_speed_max=36.0,
                wind_speed_mean=20.0,
                wind_gust_max=45.0,
                wind_direction=89.6,
            ),
            InterpolatedWeatherRecord(
                obs_date=date(1940, 1, 3),
                query_lat=lat,
                query_lon=lon,
                wind_speed_max=0.0,
                wind_speed_mean=0.0,
                wind_gust_max=0.0,
                wind_direction=90.0,
            ),
            InterpolatedWeatherRecord(
                obs_date=date(1940, 1, 4),
                query_lat=lat,
                query_lon=lon,
                wind_speed_max=None,
                wind_speed_mean=20.0,
                wind_gust_max=45.0,
                wind_direction=90.0,
            ),
        ]


class FakeBathymetryArchive:
    point_source_id = 1

    def for_point_id(self, point_id, *, n_steps):
        assert point_id == 10
        assert n_steps == 20
        return SimpleNamespace(
            get_profile=lambda direction: SimpleNamespace(
                depths_m=np.linspace(3.0, 20.0, 20)
            )
        )


def make_coastline_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(CoastlineSourceModel(id=1, name="source", points_count=1))
    session.add(
        CoastlinePointModel(id=10, source_id=1, seq=0, lon=30.0, lat=60.0)
    )
    session.add(
        CoastlineNormalSourceModel(
            id=2,
            name="normals",
            point_source_id=1,
            sea_side="right",
            tangent_delta_m=20.0,
            normals_count=1,
        )
    )
    session.add(
        CoastlineNormalModel(
            id=3,
            normal_source_id=2,
            point_id=10,
            nx=1.0,
            ny=0.0,
            normal_azimuth_deg=90.0,
        )
    )
    session.add(
        WindFetchModel(
            point_id=10,
            azimuth_deg=90.0,
            fetch_length_m=10_000,
        )
    )
    session.commit()
    return session


def test_builder_uses_interpolated_weather_and_writes_integer_power(tmp_path):
    session = make_coastline_session()
    path = tmp_path / "wave_activity.db"
    with WaveActivityRepository(path) as output:
        stats = WaveActivityDatabaseBuilder(
            coastline_session=session,
            weather_repository=FakeWeatherRepository(),
            bathymetry_archive=FakeBathymetryArchive(),
            output_repository=output,
            normal_source_id=2,
            bathy_n_steps=20,
        ).run()

    con = sqlite3.connect(path)
    point = con.execute("SELECT * FROM wave_points").fetchone()
    activity = con.execute(
        "SELECT * FROM wave_activity ORDER BY day"
    ).fetchall()
    summary = con.execute("SELECT * FROM wave_activity_summary").fetchone()
    con.close()
    session.close()

    # 3 записи погоды: 1 с wind>0 (→ power>0, в БД), 1 с wind=0 (→ power=0,
    # не в БД но в summary), 1 с wind=None (→ skipped_weather)
    assert stats.point_count == 1
    assert stats.activity_count == 1        # только день с power > 0
    assert stats.skipped_weather_rows == 1  # wind_speed_max=None
    assert stats.skipped_no_fetch_rows == 0
    assert stats.fallback_bathy_rows == 0

    assert point == (10, 30.0, 60.0, 1.0, 0.0, 90.0)

    # В wave_activity только одна запись — день с power > 0
    assert len(activity) == 1
    assert activity[0][0:3] == (10, 1, 90)  # point_id=10, day=1, direction=90
    assert isinstance(activity[0][3], int)
    assert activity[0][3] > 0

    # Summary считается по 2 дням (wind>0 + wind=0), из них 1 активный
    assert summary[0:3] == (10, 2, 1)       # point_id, n_days=2, n_active_days=1


def test_builder_rejects_bathymetry_for_another_point_source(tmp_path):
    session = make_coastline_session()
    bathymetry = FakeBathymetryArchive()
    bathymetry.point_source_id = 999
    with WaveActivityRepository(tmp_path / "wave_activity.db") as output:
        with pytest.raises(ValueError, match="different coastline point sources"):
            WaveActivityDatabaseBuilder(
                coastline_session=session,
                weather_repository=FakeWeatherRepository(),
                bathymetry_archive=bathymetry,
                output_repository=output,
                normal_source_id=2,
            ).run()
    session.close()
