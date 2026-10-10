import json
import sqlite3
from datetime import date

import numpy as np
import pytest

from src.waves.activity_indices import ActivityIndexBuilder, build_periods, export_geojson
from src.waves.activity_indices.calc import alongshore_load_wm, signed_angle_deg
from src.waves.activity_indices.constants import EPOCH, SOURCE_SCHEMA_VERSION
from src.waves.storage.schema import SCHEMA_SQL

START_DAY = 0
N_DAYS = 3 * 365 + 366 + 100
INTERVAL_YEARS = 2
MAX_ANGLE_DEG = 85.0
NORMAL_AZ = 90
ACTIVITY = [(1, 10, 120, 100), (1, 11, 60, 50), (1, 800, 90, 40), (1, 1500, 90, 70)]


def make_source(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_SQL)
    con.execute(f"PRAGMA user_version = {SOURCE_SCHEMA_VERSION}")
    con.execute("INSERT INTO wave_points VALUES (1, 30.0, 60.0, ?)", (NORMAL_AZ,))
    con.executemany("INSERT INTO wave_activity VALUES (?, ?, ?, ?)", ACTIVITY)
    con.execute(
        "INSERT INTO wave_activity_summary VALUES (1, ?, 4, 1,1,1,1,1,1,1,1,0, 0.0)",
        (N_DAYS,),
    )
    con.commit()
    con.close()


def test_periods_are_integer_days_from_epoch():
    periods = build_periods(START_DAY, N_DAYS - 1, INTERVAL_YEARS)
    assert periods[0].start_day == 0
    assert (EPOCH.replace(year=1942) - EPOCH).days == periods[1].start_day
    assert periods[0].n_days == 366 + 365
    assert not periods[-1].is_complete
    assert periods[-1].end_day == N_DAYS - 1
    assert sum(p.n_days for p in periods) == N_DAYS


def test_signed_alongshore_sign_and_cap():
    theta = signed_angle_deg(np.array([120.0, 60.0, 90.0]), NORMAL_AZ)
    assert theta.tolist() == [30.0, -30.0, 0.0]
    a = alongshore_load_wm(np.array([100.0, 100.0, 100.0]), np.array([120.0, 60.0, 90.0]), NORMAL_AZ, MAX_ANGLE_DEG)
    assert a[0] == pytest.approx(100.0 * np.tan(np.radians(30.0)))
    assert a[1] == pytest.approx(-a[0])
    assert a[2] == 0.0


def test_build_and_export(tmp_path):
    src, out, geo = tmp_path / "a.db", tmp_path / "i.db", tmp_path / "t.geojson"
    make_source(src)
    ActivityIndexBuilder(
        src, out, interval_years=INTERVAL_YEARS, start_day=START_DAY, end_day=None,
        max_angle_deg=MAX_ANGLE_DEG,
    ).run()

    con = sqlite3.connect(out)
    rows = con.execute(
        "SELECT i.normal_mean_wm, i.alongshore_net_mean_wm, i.alongshore_abs_mean_wm, "
        "i.d_normal_wm FROM point_period_indices i ORDER BY period_id"
    ).fetchall()
    first_len = 366 + 365
    t30 = float(np.tan(np.radians(30.0)))
    assert rows[0][0] == pytest.approx((100 + 50) / first_len, abs=1e-5)
    assert rows[0][1] == pytest.approx((100 * t30 - 50 * t30) / first_len, abs=1e-5)
    assert rows[0][2] == pytest.approx(150 * t30 / first_len, abs=1e-5)
    assert rows[0][3] is None and rows[1][3] is not None
    n_trend = con.execute("SELECT n_periods FROM point_trends").fetchone()[0]
    assert n_trend == con.execute("SELECT COUNT(*) FROM periods WHERE is_complete=1").fetchone()[0]
    con.close()

    for mode, expected in (("trends", 1), ("periods", 3)):
        stats = export_geojson(out, geo, mode=mode, indent=None)
        data = json.loads(geo.read_text(encoding="utf-8"))
        assert stats.feature_count == len(data["features"]) == expected
        assert data["features"][0]["geometry"]["coordinates"] == [30.0, 60.0]
    props = data["features"][0]["properties"]
    assert props["start_date"] == EPOCH.isoformat() and props["start_day"] == 0
    assert date.fromisoformat(props["end_date"]) > EPOCH


def test_rejects_days_outside_range(tmp_path):
    src = tmp_path / "a.db"
    make_source(src)
    with pytest.raises(ValueError, match="outside"):
        ActivityIndexBuilder(
            src, tmp_path / "o.db", interval_years=INTERVAL_YEARS, start_day=START_DAY, end_day=100
        ).run()
