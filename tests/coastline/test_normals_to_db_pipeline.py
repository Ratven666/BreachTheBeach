from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString, mapping
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "db_pypeline" / "2_1_normals_to_db.py"

sys.path.insert(0, str(PROJECT_ROOT))

from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
)

LAT = 54.9


def azimuth_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def write_geojson(path: Path, lines: list[LineString]) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": mapping(l)}
                    for l in lines
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'test.db').as_posix()}"


def seed_points(
    db_url: str,
    coords: list[tuple[float, float]],
    *,
    main_path: str | None,
    seq_start: int = 0,
    id_offset: int = 0,
    crs: str = "EPSG:4326",
) -> int:
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as s:
        for i in range(id_offset):
            dummy = CoastlineSourceModel(
                name=f"dummy{i}", strategy_name="x", source_mode="main",
                strategy_params="{}", main_geojson_path="x",
                other_geojson_path=None, crs="EPSG:4326", points_count=1,
            )
            s.add(dummy)
            s.flush()
            s.add(CoastlinePointModel(source_id=dummy.id, seq=0, lon=0.0, lat=0.0))
        s.commit()

        src = CoastlineSourceModel(
            name="src", strategy_name="x", source_mode="main",
            strategy_params="{}", main_geojson_path=main_path,
            other_geojson_path=None, crs=crs, points_count=len(coords),
        )
        s.add(src)
        s.flush()

        for i, (lon, lat) in enumerate(coords):
            s.add(CoastlinePointModel(
                source_id=src.id, seq=seq_start + i, lon=lon, lat=lat
            ))
        s.commit()
        return int(src.id)


def run_script(db_url: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "COASTLINE_DATABASE_URL": db_url}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


# ── Явный путь к береговой линии ──────────────────────────────────────

def test_explicit_coastline_overrides_metadata(tmp_path, db_url):
    wrong = write_geojson(
        tmp_path / "wrong.geojson", [LineString([(20.0, LAT), (20.0, LAT + 0.1)])]
    )
    right = write_geojson(
        tmp_path / "right.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(db_url, [(20.03, LAT), (20.05, LAT)], main_path=str(wrong))

    proc = run_script(
        db_url, "--coastline", str(right), "--point-source-id", str(sid),
        "--sea-side", "right", "--name", "n1",
    )
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        azimuths = s.execute(select(CoastlineNormalModel.normal_azimuth_deg)).scalars().all()

    assert len(azimuths) == 2
    for az in azimuths:
        assert azimuth_diff(az, 180.0) < AZ_TOL


AZ_TOL = 2.5


def test_metadata_path_used_when_flag_missing(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(db_url, [(20.03, LAT)], main_path=str(line))

    proc = run_script(db_url, "--point-source-id", str(sid), "--name", "n2")

    assert proc.returncode == 0, proc.stderr


def test_missing_coastline_file_fails_without_writing(tmp_path, db_url):
    sid = seed_points(db_url, [(20.03, LAT)], main_path=None)

    proc = run_script(
        db_url, "--coastline", str(tmp_path / "nope.geojson"),
        "--point-source-id", str(sid),
    )

    assert proc.returncode != 0
    with Session(create_engine(db_url)) as s:
        assert s.scalar(select(func.count()).select_from(CoastlineNormalModel)) == 0


# ── Направление линии не меняется ─────────────────────────────────────

@pytest.mark.parametrize("reverse, expected_az", [(False, 180.0), (True, 0.0)])
def test_line_direction_defines_saved_side(tmp_path, db_url, reverse, expected_az):
    coords = [(20.0, LAT), (20.1, LAT)]
    if reverse:
        coords = coords[::-1]
    line = write_geojson(tmp_path / "c.geojson", [LineString(coords)])
    sid = seed_points(db_url, [(20.03, LAT), (20.05, LAT)], main_path=str(line))

    proc = run_script(
        db_url, "--coastline", str(line), "--point-source-id", str(sid),
        "--sea-side", "right",
    )
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        azimuths = s.execute(select(CoastlineNormalModel.normal_azimuth_deg)).scalars().all()

    for az in azimuths:
        assert azimuth_diff(az, expected_az) < AZ_TOL


# ── Корректность FK ───────────────────────────────────────────────────

def test_point_id_is_real_fk_not_seq(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(
        db_url,
        [(20.02, LAT), (20.04, LAT), (20.06, LAT)],
        main_path=str(line),
        seq_start=500,
        id_offset=7,
    )

    proc = run_script(
        db_url, "--coastline", str(line), "--point-source-id", str(sid), "--name", "fk"
    )
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        point_ids = set(
            s.execute(
                select(CoastlinePointModel.id).where(CoastlinePointModel.source_id == sid)
            ).scalars()
        )
        norm_ids = s.execute(select(CoastlineNormalModel.point_id)).scalars().all()

    assert len(norm_ids) == 3
    assert set(norm_ids) == point_ids


def test_saved_normals_are_unit_and_consistent(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(db_url, [(20.02, LAT), (20.05, LAT)], main_path=str(line))

    run_script(db_url, "--coastline", str(line), "--point-source-id", str(sid))

    with Session(create_engine(db_url)) as s:
        rows = s.execute(select(CoastlineNormalModel)).scalars().all()

    assert rows
    for r in rows:
        assert math.hypot(r.nx, r.ny) == pytest.approx(1.0, abs=1e-9)
        expected = (math.degrees(math.atan2(r.nx, r.ny)) + 360.0) % 360.0
        assert azimuth_diff(r.normal_azimuth_deg, expected) < 1e-6


def test_normal_source_metadata_matches_run(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(db_url, [(20.02, LAT), (20.05, LAT)], main_path=str(line))

    run_script(
        db_url, "--coastline", str(line), "--point-source-id", str(sid),
        "--sea-side", "left", "--tangent-delta-m", "7.5", "--name", "meta",
    )

    with Session(create_engine(db_url)) as s:
        ns = s.execute(select(CoastlineNormalSourceModel)).scalars().one()

    assert ns.sea_side == "left"
    assert ns.tangent_delta_m == pytest.approx(7.5)
    assert ns.normals_count == 2
    assert ns.point_source_id == sid
    assert ns.working_crs and ns.working_crs.startswith("EPSG:")


# ── Защитные проверки ─────────────────────────────────────────────────

def test_point_far_from_reference_aborts_transaction(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(
        db_url, [(20.03, LAT), (20.05, LAT + 0.2)], main_path=str(line)
    )

    proc = run_script(
        db_url, "--coastline", str(line), "--point-source-id", str(sid),
        "--max-reference-distance-m", "500",
    )

    assert proc.returncode != 0
    with Session(create_engine(db_url)) as s:
        assert s.scalar(select(func.count()).select_from(CoastlineNormalModel)) == 0
        assert s.scalar(select(func.count()).select_from(CoastlineNormalSourceModel)) == 0


def test_non_wgs84_point_source_is_rejected(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(
        db_url, [(20.03, LAT)], main_path=str(line), crs="EPSG:32634"
    )

    proc = run_script(db_url, "--coastline", str(line), "--point-source-id", str(sid))

    assert proc.returncode != 0
    with Session(create_engine(db_url)) as s:
        assert s.scalar(select(func.count()).select_from(CoastlineNormalModel)) == 0


def test_multi_component_reference_end_to_end(tmp_path, db_url):
    long_east = LineString([(20.00, LAT), (20.10, LAT)])
    short_north = LineString([(20.30, LAT), (20.30, LAT + 0.01)])
    ref = write_geojson(tmp_path / "c.geojson", [long_east, short_north])

    sid = seed_points(
        db_url, [(20.05, LAT), (20.30, LAT + 0.005)], main_path=str(ref)
    )

    proc = run_script(
        db_url, "--coastline", str(ref), "--point-source-id", str(sid)
    )
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        rows = s.execute(
            select(CoastlineNormalModel, CoastlinePointModel.seq)
            .join(CoastlinePointModel, CoastlinePointModel.id == CoastlineNormalModel.point_id)
            .order_by(CoastlinePointModel.seq)
        ).all()

    az = [r[0].normal_azimuth_deg for r in rows]
    assert azimuth_diff(az[0], 180.0) < AZ_TOL
    assert azimuth_diff(az[1], 90.0) < AZ_TOL
