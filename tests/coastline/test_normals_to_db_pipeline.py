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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
)

LAT = 54.9

SCRIPT_TEXT = SCRIPT.read_text(encoding="utf-8")
HAS_COASTLINE_FLAG = "--coastline" in SCRIPT_TEXT

needs_coastline_flag = pytest.mark.skipif(
    not HAS_COASTLINE_FLAG,
    reason="2_1_normals_to_db.py не поддерживает --coastline",
)


def azimuth_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def write_geojson(path: Path, lines: list[LineString]) -> Path:
    path.write_text(
        json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": mapping(l)}
                for l in lines
            ],
        }),
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
    """Создаёт ровно один источник точек (скрипт берёт последний по id)."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as s:
        src = CoastlineSourceModel(
            name="src", strategy_name="x", source_mode="main",
            strategy_params="{}", main_geojson_path=main_path,
            other_geojson_path=None, crs=crs, points_count=len(coords),
        )
        s.add(src)
        s.flush()

        # Сдвигаем PK точек, чтобы id не совпадали ни с seq, ни с позицией.
        for i in range(id_offset):
            s.add(CoastlinePointModel(
                id=10_000 + i, source_id=src.id, seq=-1 - i, lon=0.0, lat=0.0
            ))
        s.flush()

        for i, (lon, lat) in enumerate(coords):
            s.add(CoastlinePointModel(
                source_id=src.id, seq=seq_start + i, lon=lon, lat=lat
            ))
        s.commit()
        return int(src.id)


def run_script(db_url: str, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "COASTLINE_DATABASE_URL": db_url,
        "PYTHONPATH": os.pathsep.join(
            [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
        ),
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=240,
    )


def count(db_url: str, model) -> int:
    with Session(create_engine(db_url)) as s:
        return s.scalar(select(func.count()).select_from(model))


# ── Базовый запуск по путям из метаданных ─────────────────────────────

def test_script_runs_with_metadata_path(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    seed_points(db_url, [(20.03, LAT), (20.05, LAT)], main_path=str(line))

    proc = run_script(db_url)

    assert proc.returncode == 0, proc.stderr
    assert count(db_url, CoastlineNormalModel) == 2
    assert count(db_url, CoastlineNormalSourceModel) == 1


def test_script_fails_without_main_geojson_path(db_url):
    seed_points(db_url, [(20.03, LAT)], main_path=None)

    proc = run_script(db_url)

    assert proc.returncode != 0
    assert count(db_url, CoastlineNormalModel) == 0


def test_saved_normals_are_unit_and_azimuth_consistent(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    seed_points(db_url, [(20.02, LAT), (20.05, LAT)], main_path=str(line))

    assert run_script(db_url).returncode == 0

    with Session(create_engine(db_url)) as s:
        rows = s.execute(select(CoastlineNormalModel)).scalars().all()

    assert rows
    for r in rows:
        assert math.hypot(r.nx, r.ny) == pytest.approx(1.0, abs=1e-9)
        expected = (math.degrees(math.atan2(r.nx, r.ny)) + 360.0) % 360.0
        assert azimuth_diff(r.normal_azimuth_deg, expected) < 1e-6
        # «right» от линии, идущей на восток: нормаль смотрит на юг.
        assert azimuth_diff(r.normal_azimuth_deg, 180.0) < 2.5


def test_normal_source_metadata(tmp_path, db_url):
    line = write_geojson(
        tmp_path / "c.geojson", [LineString([(20.0, LAT), (20.1, LAT)])]
    )
    sid = seed_points(db_url, [(20.02, LAT), (20.05, LAT)], main_path=str(line))

    assert run_script(db_url).returncode == 0

    with Session(create_engine(db_url)) as s:
        ns = s.execute(select(CoastlineNormalSourceModel)).scalars().one()

    assert ns.point_source_id == sid
    assert ns.normals_count == 2
    assert ns.sea_side == "right"
    assert ns.working_crs and ns.working_crs.startswith("EPSG:")


# ── Регрессия: FK point_id должен быть настоящим id точки ─────────────

def test_point_id_is_real_fk_not_seq(tmp_path, db_url):
    """
    Падает на старом скрипте: там seq_to_point[int(index)] использует
    позицию результата как seq. При seq_start=500 это KeyError, а при
    seq, совпадающем с позицией, но id другом, даёт чужие FK.
    """
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

    proc = run_script(db_url)
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        expected = set(s.execute(
            select(CoastlinePointModel.id).where(
                CoastlinePointModel.source_id == sid,
                CoastlinePointModel.seq >= 0,
            )
        ).scalars())
        actual = s.execute(select(CoastlineNormalModel.point_id)).scalars().all()

    assert len(actual) == 3
    assert set(actual) == expected


def test_normal_is_bound_to_correct_point_geometry(tmp_path, db_url):
    """
    Две линии с разным направлением: нормаль, привязанная к точке,
    должна соответствовать линии именно этой точки.
    """
    east = LineString([(20.00, LAT), (20.10, LAT)])
    north = LineString([(20.30, LAT), (20.30, LAT + 0.01)])
    ref = write_geojson(tmp_path / "c.geojson", [east, north])

    seed_points(
        db_url,
        [(20.05, LAT), (20.30, LAT + 0.005)],
        main_path=str(ref),
        seq_start=100,
        id_offset=3,
    )

    proc = run_script(db_url)
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        rows = s.execute(
            select(CoastlineNormalModel.normal_azimuth_deg, CoastlinePointModel.seq)
            .join(CoastlinePointModel,
                  CoastlinePointModel.id == CoastlineNormalModel.point_id)
            .order_by(CoastlinePointModel.seq)
        ).all()

    assert len(rows) == 2
    assert azimuth_diff(rows[0][0], 180.0) < 2.5     # к восточной линии
    assert azimuth_diff(rows[1][0], 90.0) < 2.5      # к северной линии


# ── Явный путь к береговой линии ──────────────────────────────────────

@needs_coastline_flag
def test_explicit_coastline_overrides_metadata(tmp_path, db_url):
    wrong = write_geojson(
        tmp_path / "wrong.geojson",
        [LineString([(20.0, LAT), (20.0, LAT + 0.1)])],
    )
    right = write_geojson(
        tmp_path / "right.geojson",
        [LineString([(20.0, LAT), (20.1, LAT)])],
    )
    sid = seed_points(db_url, [(20.03, LAT), (20.05, LAT)], main_path=str(wrong))

    proc = run_script(
        db_url, "--coastline", str(right), "--point-source-id", str(sid)
    )
    assert proc.returncode == 0, proc.stderr

    with Session(create_engine(db_url)) as s:
        azimuths = s.execute(
            select(CoastlineNormalModel.normal_azimuth_deg)
        ).scalars().all()

    assert len(azimuths) == 2
    for az in azimuths:
        assert azimuth_diff(az, 180.0) < 2.5


@needs_coastline_flag
def test_missing_explicit_coastline_writes_nothing(tmp_path, db_url):
    sid = seed_points(db_url, [(20.03, LAT)], main_path=None)

    proc = run_script(
        db_url, "--coastline", str(tmp_path / "nope.geojson"),
        "--point-source-id", str(sid),
    )

    assert proc.returncode != 0
    assert count(db_url, CoastlineNormalModel) == 0
    assert count(db_url, CoastlineNormalSourceModel) == 0
