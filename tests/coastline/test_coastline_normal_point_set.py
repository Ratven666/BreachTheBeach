# tests/coastline/domain/test_coastline_normal_point_set.py

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from src.coastline.domain.CoastlineNormalPointSet import (
    CoastlineNormalPointSet,
    CoastlineNormalsValidationReport,
)

# ──────────────────────────────────────────────────────────────────────
#  Фабрика тестового GeoDataFrame
# ──────────────────────────────────────────────────────────────────────

def _make_gdf(
    n: int = 5,
    crs: str = "EPSG:32637",
    nx_val: float = 0.0,
    ny_val: float = 1.0,
    sea_side: str = "right",
) -> gpd.GeoDataFrame:
    """Создаёт валидный GeoDataFrame с n точками береговой линии."""
    # Единичный вектор нормали: (0, 1) уже нормирован
    norm = math.hypot(nx_val, ny_val)
    nx_unit = nx_val / norm if norm else 0.0
    ny_unit = ny_val / norm if norm else 1.0

    # Единичный касательный вектор: перпендикуляр к нормали
    tx_unit = -ny_unit
    ty_unit = nx_unit

    records = []
    for i in range(n):
        x = 500_000.0 + i * 200.0   # UTM-координаты в метрах
        y = 4_800_000.0 + i * 50.0
        records.append(
            {
                "point_id":           i,
                "chainage_m":         float(i * 200),
                "tx":                 tx_unit,
                "ty":                 ty_unit,
                "nx":                 nx_unit,
                "ny":                 ny_unit,
                "normal_azimuth_deg": 90.0,
                "sea_side":           sea_side,
                "geometry":           Point(x, y),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _make_set(n: int = 5, **kwargs) -> CoastlineNormalPointSet:
    return CoastlineNormalPointSet.from_gdf(_make_gdf(n=n, **kwargs))


# ──────────────────────────────────────────────────────────────────────
#  1. Инициализация и валидация
# ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_ok(self):
        ps = _make_set()
        assert ps.count == 5
        assert ps.crs.to_epsg() == 32637

    def test_raises_on_non_geodataframe(self):
        import pandas as pd
        with pytest.raises(TypeError, match="GeoDataFrame"):
            CoastlineNormalPointSet(pd.DataFrame())

    def test_raises_on_empty_gdf(self):
        gdf = _make_gdf(n=5)
        empty = gdf.iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            CoastlineNormalPointSet(empty)

    def test_raises_on_no_crs(self):
        gdf = _make_gdf()
        gdf = gdf.set_crs(None, allow_override=True)
        with pytest.raises(ValueError, match="CRS"):
            CoastlineNormalPointSet(gdf)

    def test_raises_on_missing_columns(self):
        gdf = _make_gdf()
        gdf = gdf.drop(columns=["nx", "ny"])
        with pytest.raises(ValueError, match="missing required columns"):
            CoastlineNormalPointSet(gdf)

    def test_raises_on_non_point_geometry(self):
        gdf = _make_gdf()
        gdf["geometry"] = LineString([(0, 0), (1, 1)])
        with pytest.raises((TypeError, ValueError)):
            CoastlineNormalPointSet(gdf)

    def test_raises_on_non_numeric_column(self):
        gdf = _make_gdf()
        gdf["nx"] = "not_a_number"
        with pytest.raises(TypeError, match="numeric"):
            CoastlineNormalPointSet(gdf)

    def test_sea_side_cast_to_str(self):
        import pandas as pd
        gdf = _make_gdf()
        gdf["sea_side"] = 1  # int → должен стать str
        ps = CoastlineNormalPointSet(gdf)
        # Проверяем, что значения стали строками, а не числами
        assert pd.api.types.is_string_dtype(ps.gdf["sea_side"])

    def test_index_reset(self):
        gdf = _make_gdf(n=10).iloc[3:8]  # несброшенный индекс
        ps = CoastlineNormalPointSet(gdf)
        assert list(ps.gdf.index) == list(range(5))


# ──────────────────────────────────────────────────────────────────────
#  2. Свойства
# ──────────────────────────────────────────────────────────────────────

class TestProperties:
    def test_count(self):
        assert _make_set(7).count == 7

    def test_empty_false(self):
        assert not _make_set().empty

    def test_bounds_type(self):
        bounds = _make_set().bounds
        assert len(bounds) == 4
        assert all(isinstance(v, float) for v in bounds)

    def test_sea_side_single(self):
        ps = _make_set(sea_side="left")
        assert ps.sea_side == "left"

    def test_sea_side_multiple(self):
        gdf = _make_gdf(n=4)
        gdf.loc[0:1, "sea_side"] = "left"
        gdf.loc[2:3, "sea_side"] = "right"
        ps = CoastlineNormalPointSet(gdf)
        assert "left" in ps.sea_side
        assert "right" in ps.sea_side


# ──────────────────────────────────────────────────────────────────────
#  3. sort_by_chainage
# ──────────────────────────────────────────────────────────────────────

class TestSortByChainage:
    def test_ascending(self):
        ps = _make_set(5)
        ps_sorted = ps.sort_by_chainage(ascending=True)
        chainages = ps_sorted.gdf["chainage_m"].tolist()
        assert chainages == sorted(chainages)

    def test_descending(self):
        ps = _make_set(5)
        ps_sorted = ps.sort_by_chainage(ascending=False)
        chainages = ps_sorted.gdf["chainage_m"].tolist()
        assert chainages == sorted(chainages, reverse=True)

    def test_returns_new_object(self):
        ps = _make_set()
        ps2 = ps.sort_by_chainage()
        assert ps is not ps2


# ──────────────────────────────────────────────────────────────────────
#  4. subset_by_chainage
# ──────────────────────────────────────────────────────────────────────

class TestSubsetByChainage:
    def test_start_only(self):
        ps = _make_set(10)
        sub = ps.subset_by_chainage(start_m=400.0)
        assert sub.gdf["chainage_m"].min() >= 400.0

    def test_end_only(self):
        ps = _make_set(10)
        sub = ps.subset_by_chainage(end_m=600.0)
        assert sub.gdf["chainage_m"].max() <= 600.0

    def test_start_and_end(self):
        ps = _make_set(10)
        sub = ps.subset_by_chainage(start_m=200.0, end_m=600.0)
        assert sub.gdf["chainage_m"].min() >= 200.0
        assert sub.gdf["chainage_m"].max() <= 600.0

    def test_empty_result_raises(self):
        ps = _make_set(5)
        with pytest.raises(ValueError, match="empty"):
            ps.subset_by_chainage(start_m=99_999.0)


# ──────────────────────────────────────────────────────────────────────
#  5. to_normal_lines_gdf
# ──────────────────────────────────────────────────────────────────────

class TestToNormalLinesGdf:
    def test_returns_geodataframe(self):
        ps = _make_set()
        lines = ps.to_normal_lines_gdf(normal_length_m=100.0)
        assert isinstance(lines, gpd.GeoDataFrame)

    def test_row_count_matches(self):
        ps = _make_set(7)
        lines = ps.to_normal_lines_gdf(normal_length_m=100.0)
        assert len(lines) == 7

    def test_geometry_is_linestring(self):
        ps = _make_set()
        lines = ps.to_normal_lines_gdf(normal_length_m=100.0)
        assert all(isinstance(g, LineString) for g in lines.geometry)

    def test_line_length_correct(self):
        """Длина LineString в метрах должна совпадать с normal_length_m."""
        ps = _make_set(nx_val=0.0, ny_val=1.0)
        target = 300.0
        lines = ps.to_normal_lines_gdf(normal_length_m=target)
        for geom in lines.geometry:
            assert abs(geom.length - target) < 1e-6

    def test_crs_preserved(self):
        """to_normal_lines_gdf должен сохранять метрическую CRS объекта."""
        ps = _make_set(crs="EPSG:32637")
        lines = ps.to_normal_lines_gdf(normal_length_m=100.0)
        assert lines.crs.to_epsg() == 32637

    def test_invalid_length_raises(self):
        ps = _make_set()
        with pytest.raises(ValueError, match="must be > 0"):
            ps.to_normal_lines_gdf(normal_length_m=0.0)

    def test_negative_length_raises(self):
        ps = _make_set()
        with pytest.raises(ValueError):
            ps.to_normal_lines_gdf(normal_length_m=-50.0)

    def test_normal_length_m_column_present(self):
        ps = _make_set()
        lines = ps.to_normal_lines_gdf(normal_length_m=200.0)
        assert "normal_length_m" in lines.columns
        assert (lines["normal_length_m"] == 200.0).all()

    def test_direction_correct(self):
        """Конец линии смещён строго в направлении (nx, ny)."""
        ps = _make_set(nx_val=1.0, ny_val=0.0)  # нормаль вдоль оси X
        lines = ps.to_normal_lines_gdf(normal_length_m=500.0)
        for i, row in lines.iterrows():
            coords = list(row.geometry.coords)
            dx = coords[1][0] - coords[0][0]
            dy = coords[1][1] - coords[0][1]
            assert abs(dx - 500.0) < 1e-6
            assert abs(dy) < 1e-6


# ──────────────────────────────────────────────────────────────────────
#  6. to_geojson — координаты в WGS84
# ──────────────────────────────────────────────────────────────────────

class TestToGeoJson:
    def test_output_crs_is_wgs84(self, tmp_path):
        ps = _make_set()
        out = tmp_path / "points.geojson"
        ps.to_geojson(out)
        loaded = gpd.read_file(out)
        assert loaded.crs.to_epsg() == 4326

    def test_coordinates_in_degree_range(self, tmp_path):
        """lon в [-180, 180], lat в [-90, 90]."""
        ps = _make_set()
        out = tmp_path / "points.geojson"
        ps.to_geojson(out)
        loaded = gpd.read_file(out)
        for geom in loaded.geometry:
            assert -180 <= geom.x <= 180
            assert -90  <= geom.y <= 90

    def test_count_preserved(self, tmp_path):
        ps = _make_set(8)
        out = tmp_path / "pts.geojson"
        ps.to_geojson(out)
        loaded = gpd.read_file(out)
        assert len(loaded) == 8

    def test_creates_parent_dirs(self, tmp_path):
        ps = _make_set()
        out = tmp_path / "deep" / "nested" / "points.geojson"
        ps.to_geojson(out)
        assert out.exists()


# ──────────────────────────────────────────────────────────────────────
#  7. export_normal_lines_geojson — координаты в WGS84
# ──────────────────────────────────────────────────────────────────────

class TestExportNormalLinesGeoJson:
    def test_output_crs_is_wgs84(self, tmp_path):
        ps = _make_set()
        out = tmp_path / "lines.geojson"
        ps.export_normal_lines_geojson(out, normal_length_m=100.0)
        loaded = gpd.read_file(out)
        assert loaded.crs.to_epsg() == 4326

    def test_geometry_type(self, tmp_path):
        ps = _make_set()
        out = tmp_path / "lines.geojson"
        ps.export_normal_lines_geojson(out, normal_length_m=100.0)
        loaded = gpd.read_file(out)
        assert all(g.geom_type == "LineString" for g in loaded.geometry)

    def test_coordinates_in_degree_range(self, tmp_path):
        ps = _make_set()
        out = tmp_path / "lines.geojson"
        ps.export_normal_lines_geojson(out, normal_length_m=100.0)
        loaded = gpd.read_file(out)
        for geom in loaded.geometry:
            for lon, lat in geom.coords:
                assert -180 <= lon <= 180
                assert -90  <= lat <= 90

    def test_count_preserved(self, tmp_path):
        ps = _make_set(6)
        out = tmp_path / "lines.geojson"
        ps.export_normal_lines_geojson(out, normal_length_m=200.0)
        loaded = gpd.read_file(out)
        assert len(loaded) == 6


# ──────────────────────────────────────────────────────────────────────
#  8. to_gpkg — CRS метрическая (без репроецирования)
# ──────────────────────────────────────────────────────────────────────

class TestToGpkg:
    def test_crs_preserved(self, tmp_path):
        ps = _make_set(crs="EPSG:32637")
        out = tmp_path / "points.gpkg"
        ps.to_gpkg(out, layer="test_layer")
        loaded = gpd.read_file(out, layer="test_layer")
        assert loaded.crs.to_epsg() == 32637

    def test_count_preserved(self, tmp_path):
        ps = _make_set(9)
        out = tmp_path / "points.gpkg"
        ps.to_gpkg(out)
        loaded = gpd.read_file(out)
        assert len(loaded) == 9


# ──────────────────────────────────────────────────────────────────────
#  9. validate_vectors
# ──────────────────────────────────────────────────────────────────────

class TestValidateVectors:
    def test_valid_set(self):
        ps = _make_set()
        report = ps.validate_vectors()
        assert isinstance(report, CoastlineNormalsValidationReport)
        assert report.is_valid

    def test_invalid_normal_detected(self):
        gdf = _make_gdf()
        gdf.loc[0, "nx"] = 5.0  # не единичный вектор
        gdf.loc[0, "ny"] = 5.0
        ps = CoastlineNormalPointSet(gdf)
        report = ps.validate_vectors()
        assert report.invalid_normal_count > 0
        assert not report.is_valid

    def test_invalid_azimuth_detected(self):
        gdf = _make_gdf()
        gdf.loc[0, "normal_azimuth_deg"] = 400.0  # вне [0, 360)
        ps = CoastlineNormalPointSet(gdf)
        report = ps.validate_vectors()
        assert report.invalid_azimuth_count > 0

    def test_chainage_not_sorted_detected(self):
        gdf = _make_gdf(n=5)
        gdf = gdf.iloc[::-1].reset_index(drop=True)  # обратный порядок
        ps = CoastlineNormalPointSet(gdf)
        report = ps.validate_vectors()
        assert report.chainage_not_sorted

    def test_duplicated_point_ids(self):
        gdf = _make_gdf(n=5)
        gdf.loc[1, "point_id"] = gdf.loc[0, "point_id"]  # дубликат
        ps = CoastlineNormalPointSet(gdf)
        report = ps.validate_vectors()
        assert report.duplicated_point_ids > 0


# ──────────────────────────────────────────────────────────────────────
#  10. summary
# ──────────────────────────────────────────────────────────────────────

class TestSummary:
    def test_count(self):
        ps = _make_set(6)
        assert ps.summary().count == 6

    def test_chainage_range(self):
        ps = _make_set(5)
        s = ps.summary()
        assert s.min_chainage_m == 0.0
        assert s.max_chainage_m == 800.0  # (5-1)*200

    def test_crs_string(self):
        ps = _make_set(crs="EPSG:32637")
        assert "32637" in ps.summary().crs

    def test_sea_side(self):
        ps = _make_set(sea_side="left")
        assert ps.summary().sea_side == "left"


# ──────────────────────────────────────────────────────────────────────
#  11. copy / to_crs
# ──────────────────────────────────────────────────────────────────────

class TestCopyAndToCrs:
    def test_copy_independent(self):
        ps = _make_set()
        ps2 = ps.copy()
        ps2.gdf.loc[0, "chainage_m"] = -999
        assert ps.gdf.loc[0, "chainage_m"] != -999

    def test_to_crs_changes_crs(self):
        ps = _make_set(crs="EPSG:32637")
        ps4326 = ps.to_crs("EPSG:4326")
        assert ps4326.crs.to_epsg() == 4326

    def test_to_crs_original_unchanged(self):
        ps = _make_set(crs="EPSG:32637")
        ps.to_crs("EPSG:4326")
        assert ps.crs.to_epsg() == 32637
