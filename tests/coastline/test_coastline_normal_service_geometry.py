from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from loguru import logger
from shapely.geometry import LineString, Point, mapping

from src.coastline.domain.CoastlineDataset import CoastlineDataset
from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
from src.coastline.domain.CoastlinePointSet import CoastlinePointSet, PointSetMeta
from src.coastline.services.CoastlineNormalService import (
    CoastlineNormalConfig,
    CoastlineNormalService,
)

LAT = 54.9
# Сближение меридианов между WGS84 и UTM даёт около 1 градуса.
AZ_TOL = 2.5


# ── Вспомогательные функции ───────────────────────────────────────────

def write_geojson(path: Path, lines: list[LineString]) -> Path:
    features = [
        {"type": "Feature", "properties": {}, "geometry": mapping(line)}
        for line in lines
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return path


def make_dataset(path: Path, lines: list[LineString]) -> CoastlineDataset:
    write_geojson(path, lines)
    return CoastlineDataset.from_geojson(
        main_path=path, other_path=None, name="test"
    )


def make_point_set(
    coords: list[tuple[float, float]],
    *,
    seq: list[int] | None = None,
    point_ids: list[int] | None = None,
) -> CoastlinePointSet:
    n = len(coords)
    gdf = gpd.GeoDataFrame(
        {
            "seq": seq if seq is not None else list(range(n)),
            "point_id": (
                point_ids if point_ids is not None
                else [1000 + i for i in range(n)]
            ),
            "geometry": [Point(x, y) for x, y in coords],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return CoastlinePointSet(
        gdf=gdf,
        meta=PointSetMeta(
            name="test",
            source_dataset_name="",
            strategy_name="",
            source_mode="",
            points_count=n,
        ),
    )


def azimuth_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build(dataset, point_set, **cfg) -> CoastlineNormalPointSet:
    params = dict(
        sea_side="right",
        tangent_delta_m=10.0,
        working_crs=str(dataset.metric_crs),
        max_reference_distance_m=500.0,
    )
    params.update(cfg)
    return CoastlineNormalService(
        CoastlineNormalConfig(**params)
    ).build_points_with_normals(
        point_set=point_set, dataset=dataset, name="t"
    )


@pytest.fixture
def east_line() -> LineString:
    return LineString([(20.00, LAT), (20.10, LAT)])


@pytest.fixture
def west_line(east_line) -> LineString:
    return LineString(list(east_line.coords)[::-1])


# ── Конфигурация ──────────────────────────────────────────────────────

def test_config_normalizes_and_validates():
    assert CoastlineNormalConfig(sea_side="LEFT").sea_side == "left"

    with pytest.raises(ValueError):
        CoastlineNormalConfig(sea_side="up")
    with pytest.raises(ValueError):
        CoastlineNormalConfig(tangent_delta_m=0)
    with pytest.raises(ValueError):
        CoastlineNormalConfig(normal_length_m=-1)
    with pytest.raises(ValueError):
        CoastlineNormalConfig(max_reference_distance_m=-1)

    assert CoastlineNormalConfig(max_reference_distance_m=None)


# ── Базовая геометрия ─────────────────────────────────────────────────

def test_right_normal_of_eastward_line_points_south(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.03, LAT + 0.0002), (20.05, LAT + 0.0002)])

    for az in build(ds, ps, sea_side="right").gdf["normal_azimuth_deg"]:
        assert azimuth_diff(az, 180.0) < AZ_TOL


def test_left_normal_of_eastward_line_points_north(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.03, LAT), (20.05, LAT)])

    for az in build(ds, ps, sea_side="left").gdf["normal_azimuth_deg"]:
        assert azimuth_diff(az, 0.0) < AZ_TOL


def test_left_and_right_are_opposite(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.03, LAT), (20.06, LAT)])

    left = build(ds, ps, sea_side="left").gdf
    right = build(ds, ps, sea_side="right").gdf

    np.testing.assert_allclose(left["nx"], -right["nx"], atol=1e-9)
    np.testing.assert_allclose(left["ny"], -right["ny"], atol=1e-9)


def test_normal_is_perpendicular_and_unit_on_curved_line(tmp_path):
    a = np.linspace(0, math.pi / 2, 200)
    curve = LineString(
        [(20.0 + 0.05 * math.cos(t), LAT + 0.03 * math.sin(t)) for t in a]
    )
    ds = make_dataset(tmp_path / "c.geojson", [curve])
    pts = [curve.interpolate(f, normalized=True) for f in (0.1, 0.3, 0.5, 0.8)]
    ps = make_point_set([(p.x, p.y) for p in pts])

    g = build(ds, ps).gdf

    assert np.abs(g["tx"] * g["nx"] + g["ty"] * g["ny"]).max() < 1e-9
    np.testing.assert_allclose(np.hypot(g["nx"], g["ny"]), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.hypot(g["tx"], g["ty"]), 1.0, atol=1e-9)


def test_azimuth_matches_vector(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.04, LAT)])

    row = build(ds, ps).gdf.iloc[0]
    expected = (math.degrees(math.atan2(row["nx"], row["ny"])) + 360.0) % 360.0

    assert azimuth_diff(row["normal_azimuth_deg"], expected) < 1e-9


def test_normal_never_collinear_with_tangent_on_zigzag(tmp_path):
    zigzag = LineString(
        [(20.00, LAT), (20.01, LAT + 0.004), (20.02, LAT),
         (20.03, LAT + 0.004), (20.04, LAT)]
    )
    ds = make_dataset(tmp_path / "c.geojson", [zigzag])
    pts = [zigzag.interpolate(f, normalized=True) for f in np.linspace(0.05, 0.95, 15)]
    ps = make_point_set([(p.x, p.y) for p in pts])

    g = build(ds, ps, tangent_delta_m=5.0).gdf

    assert np.abs(g["tx"] * g["nx"] + g["ty"] * g["ny"]).max() < 1e-9


def test_tangent_follows_line_vertex_order(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT)])

    row = build(ds, ps).gdf.iloc[0]

    assert row["tx"] > 0.99          # линия идёт на восток


# ── Направление линии определяет сторону ──────────────────────────────

def test_reversed_line_flips_normal(tmp_path, east_line, west_line):
    coords = [(20.03, LAT), (20.06, LAT)]

    a = build(make_dataset(tmp_path / "e.geojson", [east_line]),
              make_point_set(coords)).gdf
    b = build(make_dataset(tmp_path / "w.geojson", [west_line]),
              make_point_set(coords)).gdf

    np.testing.assert_allclose(a["nx"], -b["nx"], atol=1e-6)
    np.testing.assert_allclose(a["ny"], -b["ny"], atol=1e-6)
    assert azimuth_diff(a["normal_azimuth_deg"].iloc[0], 180.0) < AZ_TOL
    assert azimuth_diff(b["normal_azimuth_deg"].iloc[0], 0.0) < AZ_TOL


def test_point_order_does_not_change_normals(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    coords = [(20.02, LAT), (20.04, LAT), (20.06, LAT)]

    fwd = build(ds, make_point_set(coords)).gdf.set_index("point_id")
    rev = build(
        ds,
        make_point_set(coords[::-1], point_ids=[1002, 1001, 1000], seq=[0, 1, 2]),
    ).gdf.set_index("point_id")

    for pid in (1000, 1001, 1002):
        assert fwd.loc[pid, "nx"] == pytest.approx(rev.loc[pid, "nx"], abs=1e-9)
        assert fwd.loc[pid, "ny"] == pytest.approx(rev.loc[pid, "ny"], abs=1e-9)


def test_line_direction_is_not_altered_by_service(tmp_path):
    # Две соседние линии с противоположным направлением не склеиваются
    # и не разворачиваются: каждая сохраняет свою ориентацию.
    east = LineString([(20.00, LAT), (20.05, LAT)])
    west = LineString([(20.10, LAT + 0.01), (20.06, LAT + 0.01)])
    ds = make_dataset(tmp_path / "c.geojson", [east, west])
    ps = make_point_set([(20.02, LAT), (20.08, LAT + 0.01)])

    g = build(ds, ps).gdf.sort_values("seq")

    assert g["reference_part_id"].nunique() == 2
    assert azimuth_diff(g.iloc[0]["normal_azimuth_deg"], 180.0) < AZ_TOL
    assert azimuth_diff(g.iloc[1]["normal_azimuth_deg"], 0.0) < AZ_TOL


def test_mixed_point_order_only_logs_warning(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set(
        [(20.02, LAT), (20.08, LAT), (20.04, LAT)], seq=[0, 1, 2]
    )

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        res = build(ds, ps)
    finally:
        logger.remove(sink_id)

    assert any("not monotonic" in m for m in messages)
    # Данные не изменены: направление линии не затронуто.
    assert res.gdf["tx"].min() > 0.99


# ── Несколько компонентов ─────────────────────────────────────────────

def test_all_components_are_used_not_only_longest(tmp_path):
    long_east = LineString([(20.00, LAT), (20.10, LAT)])
    short_north = LineString([(20.30, LAT), (20.30, LAT + 0.01)])
    ds = make_dataset(tmp_path / "c.geojson", [long_east, short_north])
    ps = make_point_set([(20.05, LAT), (20.30, LAT + 0.005)])

    g = build(ds, ps).gdf.sort_values("seq")

    assert azimuth_diff(g.iloc[0]["normal_azimuth_deg"], 180.0) < AZ_TOL
    assert azimuth_diff(g.iloc[1]["normal_azimuth_deg"], 90.0) < AZ_TOL
    assert g.iloc[0]["reference_part_id"] != g.iloc[1]["reference_part_id"]


def test_multilinestring_feature_is_split_into_parts(tmp_path):
    from shapely.geometry import MultiLineString

    multi = MultiLineString([
        [(20.00, LAT), (20.05, LAT)],
        [(20.20, LAT), (20.20, LAT + 0.01)],
    ])
    path = tmp_path / "m.geojson"
    path.write_text(
        json.dumps({
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {},
                          "geometry": mapping(multi)}],
        }),
        encoding="utf-8",
    )
    ds = CoastlineDataset.from_geojson(main_path=path, other_path=None, name="m")
    ps = make_point_set([(20.02, LAT), (20.20, LAT + 0.005)])

    g = build(ds, ps).gdf

    assert g["reference_part_id"].nunique() == 2


# ── Контроль расстояния ───────────────────────────────────────────────

def test_point_beyond_max_distance_raises(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT + 0.1)])     # около 11 км

    with pytest.raises(ValueError, match="nearest reference coastline"):
        build(ds, ps, max_reference_distance_m=500.0)


def test_max_distance_none_disables_check(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT + 0.1)])

    assert len(build(ds, ps, max_reference_distance_m=None).gdf) == 1


def test_reference_distance_is_reported(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT + 0.0009)])   # около 100 м

    d = build(ds, ps).gdf.iloc[0]["reference_distance_m"]

    assert 90.0 < d < 110.0


# ── Идентификаторы ────────────────────────────────────────────────────

def test_point_id_from_input_is_preserved(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.04, LAT), (20.06, LAT)])

    assert build(ds, ps).gdf["point_id"].tolist() == [1000, 1001, 1002]


def test_point_id_does_not_depend_on_gdf_index(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.04, LAT)])
    ps.gdf.index = [50, 77]

    assert build(ds, ps).gdf["point_id"].tolist() == [1000, 1001]


def test_point_id_falls_back_to_position_when_missing(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.04, LAT)])
    ps.gdf = ps.gdf.drop(columns=["point_id"])

    assert build(ds, ps).gdf["point_id"].tolist() == [0, 1]


def test_result_index_is_reset_not_seq(tmp_path, east_line):
    # Индекс результата — позиция 0..n-1. Использовать его как seq нельзя.
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set(
        [(20.02, LAT), (20.04, LAT)], seq=[500, 501], point_ids=[7, 8]
    )

    g = build(ds, ps).gdf

    assert g.index.tolist() == [0, 1]
    assert g["seq"].tolist() == [500, 501]
    assert g["point_id"].tolist() == [7, 8]


# ── Валидатор ─────────────────────────────────────────────────────────

def test_validator_accepts_correct_result(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.05, LAT)])

    report = build(ds, ps).validate_vectors()

    assert report.is_valid
    assert report.max_abs_dot_product < 1e-9


def test_chainage_not_sorted_is_informational_only(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.06, LAT), (20.02, LAT), (20.04, LAT)])

    report = build(ds, ps).validate_vectors()

    assert report.chainage_not_sorted is True
    assert report.is_valid is True


def _corrupt(res: CoastlineNormalPointSet, **changes) -> CoastlineNormalPointSet:
    bad = res.gdf.copy()
    for column, value in changes.items():
        bad[column] = value
    return CoastlineNormalPointSet(bad, name="bad")


def test_validator_detects_normal_along_tangent(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.02, LAT), (20.05, LAT)]))

    nx, ny = res.gdf["tx"], res.gdf["ty"]
    az = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    report = _corrupt(res, nx=nx, ny=ny, normal_azimuth_deg=az).validate_vectors()

    assert not report.is_valid
    assert report.invalid_orthogonality_count == len(res.gdf)


def test_validator_detects_wrong_side(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.02, LAT), (20.05, LAT)]), sea_side="right")

    report = _corrupt(
        res,
        nx=-res.gdf["nx"],
        ny=-res.gdf["ny"],
        normal_azimuth_deg=(res.gdf["normal_azimuth_deg"] + 180.0) % 360.0,
    ).validate_vectors()

    assert report.invalid_side_count == len(res.gdf)
    assert not report.is_valid


def test_validator_detects_inconsistent_azimuth(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.02, LAT)]))

    report = _corrupt(
        res, normal_azimuth_deg=(res.gdf["normal_azimuth_deg"] + 30.0) % 360.0
    ).validate_vectors()

    assert report.invalid_azimuth_consistency_count == 1
    assert not report.is_valid


def test_validator_detects_duplicate_point_ids(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.02, LAT), (20.04, LAT)]))

    report = _corrupt(res, point_id=[5, 5]).validate_vectors()

    assert report.duplicated_point_ids == 1
    assert not report.is_valid


def test_normal_set_rejects_missing_columns(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.02, LAT)]))

    with pytest.raises(ValueError, match="missing required columns"):
        CoastlineNormalPointSet(res.gdf.drop(columns=["nx"]), name="x")


# ── Линии нормалей ────────────────────────────────────────────────────

def test_normal_lines_have_requested_length_and_direction(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.05, LAT)]))

    line = res.to_normal_lines_gdf(normal_length_m=123.0).geometry.iloc[0]
    row = res.gdf.iloc[0]
    (x0, y0), (x1, y1) = line.coords

    assert line.length == pytest.approx(123.0, abs=1e-6)
    assert (x1 - x0) / 123.0 == pytest.approx(row["nx"], abs=1e-9)
    assert (y1 - y0) / 123.0 == pytest.approx(row["ny"], abs=1e-9)


def test_service_build_normal_lines_uses_config_length(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT)])
    svc = CoastlineNormalService(
        CoastlineNormalConfig(
            normal_length_m=250.0,
            tangent_delta_m=10.0,
            working_crs=str(ds.metric_crs),
        )
    )

    lines = svc.build_normal_lines(ps, ds)

    assert lines.geometry.iloc[0].length == pytest.approx(250.0, abs=1e-6)


def test_normal_lines_refuse_geographic_crs(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    res = build(ds, make_point_set([(20.05, LAT)])).to_crs("EPSG:4326")

    with pytest.raises(ValueError, match="geographic"):
        res.to_normal_lines_gdf(100.0)
