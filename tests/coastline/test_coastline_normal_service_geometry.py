from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, mapping

from src.coastline.domain.CoastlineDataset import CoastlineDataset
from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
from src.coastline.domain.CoastlinePointSet import CoastlinePointSet, PointSetMeta
from src.coastline.services.CoastlineNormalService import (
    CoastlineNormalConfig,
    CoastlineNormalService,
)

LAT = 54.9
# Допуск на сближение меридианов между WGS84 и UTM (около 1 градуса).
AZ_TOL = 2.5


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


def make_point_set(coords: list[tuple[float, float]]) -> CoastlinePointSet:
    gdf = gpd.GeoDataFrame(
        {
            "seq": list(range(len(coords))),
            "point_id": [1000 + i for i in range(len(coords))],
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
            points_count=len(gdf),
        ),
    )


def azimuth_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build(dataset, point_set, **cfg):
    params = dict(
        sea_side="right",
        tangent_delta_m=10.0,
        working_crs=str(dataset.metric_crs),
        max_reference_distance_m=500.0,
    )
    params.update(cfg)
    return CoastlineNormalService(
        CoastlineNormalConfig(**params)
    ).build_points_with_normals(point_set=point_set, dataset=dataset, name="t")


@pytest.fixture
def east_line() -> LineString:
    return LineString([(20.00, LAT), (20.10, LAT)])


# ── Базовая геометрия ─────────────────────────────────────────────────

def test_right_normal_of_eastward_line_points_south(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.03, LAT + 0.0002), (20.05, LAT + 0.0002)])

    res = build(ds, ps, sea_side="right")

    for az in res.gdf["normal_azimuth_deg"]:
        assert azimuth_diff(az, 180.0) < AZ_TOL


def test_left_normal_of_eastward_line_points_north(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.03, LAT), (20.05, LAT)])

    res = build(ds, ps, sea_side="left")

    for az in res.gdf["normal_azimuth_deg"]:
        assert azimuth_diff(az, 0.0) < AZ_TOL


def test_left_and_right_are_opposite(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.03, LAT), (20.06, LAT)])

    left = build(ds, ps, sea_side="left").gdf
    right = build(ds, ps, sea_side="right").gdf

    np.testing.assert_allclose(left["nx"], -right["nx"], atol=1e-9)
    np.testing.assert_allclose(left["ny"], -right["ny"], atol=1e-9)


def test_normal_is_perpendicular_and_unit_on_curved_line(tmp_path):
    t = np.linspace(0, math.pi / 2, 200)
    curve = LineString(
        [(20.0 + 0.05 * math.cos(a), LAT + 0.03 * math.sin(a)) for a in t]
    )
    ds = make_dataset(tmp_path / "c.geojson", [curve])
    pts = [curve.interpolate(f, normalized=True) for f in (0.1, 0.3, 0.5, 0.8)]
    ps = make_point_set([(p.x, p.y) for p in pts])

    g = build(ds, ps).gdf

    assert np.abs(g["tx"] * g["nx"] + g["ty"] * g["ny"]).max() < 1e-9
    np.testing.assert_allclose(np.hypot(g["nx"], g["ny"]), 1.0, atol=1e-9)


def test_azimuth_matches_vector(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.04, LAT)])

    row = build(ds, ps).gdf.iloc[0]
    expected = (math.degrees(math.atan2(row["nx"], row["ny"])) + 360.0) % 360.0

    assert azimuth_diff(row["normal_azimuth_deg"], expected) < 1e-9


def test_normal_never_collinear_with_tangent(tmp_path):
    zigzag = LineString(
        [(20.00, LAT), (20.01, LAT + 0.004), (20.02, LAT),
         (20.03, LAT + 0.004), (20.04, LAT)]
    )
    ds = make_dataset(tmp_path / "c.geojson", [zigzag])
    pts = [zigzag.interpolate(f, normalized=True) for f in np.linspace(0.05, 0.95, 15)]
    ps = make_point_set([(p.x, p.y) for p in pts])

    g = build(ds, ps, tangent_delta_m=5.0).gdf

    assert np.abs(g["tx"] * g["nx"] + g["ty"] * g["ny"]).max() < 1e-9


# ── Направление линии не меняется ─────────────────────────────────────

@pytest.mark.parametrize(
    "reverse, side, expected_az",
    [
        (False, "right", 180.0),
        (False, "left", 0.0),
        (True, "right", 0.0),
        (True, "left", 180.0),
    ],
)
def test_direction_of_geometry_is_never_changed(
    tmp_path, east_line, reverse, side, expected_az
):
    line = LineString(list(east_line.coords)[::-1]) if reverse else east_line
    ds = make_dataset(tmp_path / "c.geojson", [line])

    # Порядок точек намеренно противоположен геометрии.
    ps = make_point_set([(20.06, LAT), (20.04, LAT), (20.02, LAT)])

    g = build(ds, ps, sea_side=side).gdf

    for az in g["normal_azimuth_deg"]:
        assert azimuth_diff(az, expected_az) < AZ_TOL


def test_point_order_does_not_affect_result(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])

    fwd = build(ds, make_point_set([(20.02, LAT), (20.04, LAT)])).gdf
    bwd = build(ds, make_point_set([(20.04, LAT), (20.02, LAT)])).gdf

    assert fwd["normal_azimuth_deg"].round(6).tolist() == pytest.approx(
        bwd["normal_azimuth_deg"].round(6).tolist()[::-1]
    )


def test_components_with_opposite_directions_keep_own_side(tmp_path):
    east = LineString([(20.00, LAT), (20.10, LAT)])
    west = LineString([(20.30, LAT), (20.20, LAT)])   # идёт на запад
    ds = make_dataset(tmp_path / "c.geojson", [east, west])
    ps = make_point_set([(20.05, LAT), (20.25, LAT)])

    g = build(ds, ps).gdf.sort_values("seq")

    assert azimuth_diff(g.iloc[0]["normal_azimuth_deg"], 180.0) < AZ_TOL
    assert azimuth_diff(g.iloc[1]["normal_azimuth_deg"], 0.0) < AZ_TOL


def test_touching_segments_are_not_merged(tmp_path):
    a = LineString([(20.00, LAT), (20.05, LAT)])
    b = LineString([(20.05, LAT), (20.10, LAT)])
    ds = make_dataset(tmp_path / "c.geojson", [a, b])
    ps = make_point_set([(20.02, LAT), (20.08, LAT)])

    g = build(ds, ps).gdf

    assert g["reference_part_id"].nunique() == 2


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


def test_point_beyond_max_distance_raises(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT + 0.1)])

    with pytest.raises(ValueError, match="nearest reference coastline"):
        build(ds, ps, max_reference_distance_m=500.0)


def test_max_distance_none_disables_check(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT + 0.1)])

    assert len(build(ds, ps, max_reference_distance_m=None).gdf) == 1


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


# ── Валидатор ─────────────────────────────────────────────────────────

def test_validator_accepts_correct_result(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.05, LAT)])

    report = build(ds, ps).validate_vectors()

    assert report.is_valid
    assert report.max_abs_dot_product < 1e-9


def test_unsorted_chainage_does_not_invalidate(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.06, LAT), (20.04, LAT), (20.02, LAT)])

    report = build(ds, ps).validate_vectors()

    assert report.chainage_not_sorted is True
    assert report.is_valid


def test_validator_detects_normal_along_tangent(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.05, LAT)])
    bad = build(ds, ps).gdf.copy()

    bad["nx"], bad["ny"] = bad["tx"], bad["ty"]
    bad["normal_azimuth_deg"] = (
        np.degrees(np.arctan2(bad["nx"], bad["ny"])) + 360.0
    ) % 360.0

    report = CoastlineNormalPointSet(bad, name="bad").validate_vectors()

    assert not report.is_valid
    assert report.invalid_orthogonality_count == len(bad)


def test_validator_detects_wrong_side(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT), (20.05, LAT)])
    bad = build(ds, ps, sea_side="right").gdf.copy()

    bad["nx"] *= -1.0
    bad["ny"] *= -1.0
    bad["normal_azimuth_deg"] = (bad["normal_azimuth_deg"] + 180.0) % 360.0

    report = CoastlineNormalPointSet(bad, name="bad").validate_vectors()

    assert report.invalid_side_count == len(bad)
    assert not report.is_valid


def test_validator_detects_inconsistent_azimuth(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.02, LAT)])
    bad = build(ds, ps).gdf.copy()

    bad["normal_azimuth_deg"] = (bad["normal_azimuth_deg"] + 30.0) % 360.0

    report = CoastlineNormalPointSet(bad, name="bad").validate_vectors()

    assert report.invalid_azimuth_consistency_count == 1
    assert not report.is_valid


# ── Линии нормалей ────────────────────────────────────────────────────

def test_normal_lines_have_requested_length_and_direction(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT)])

    res = build(ds, ps)
    line = res.to_normal_lines_gdf(normal_length_m=123.0).geometry.iloc[0]

    assert line.length == pytest.approx(123.0, abs=1e-6)

    row = res.gdf.iloc[0]
    (x0, y0), (x1, y1) = line.coords
    assert (x1 - x0) / 123.0 == pytest.approx(row["nx"], abs=1e-9)
    assert (y1 - y0) / 123.0 == pytest.approx(row["ny"], abs=1e-9)


def test_normal_lines_refuse_geographic_crs(tmp_path, east_line):
    ds = make_dataset(tmp_path / "c.geojson", [east_line])
    ps = make_point_set([(20.05, LAT)])
    res = build(ds, ps).to_crs("EPSG:4326")

    with pytest.raises(ValueError, match="geographic"):
        res.to_normal_lines_gdf(100.0)
