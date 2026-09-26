from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from geopandas import GeoDataFrame
from shapely.geometry import Point


def _make_meta(
    source_dataset_name: str = "test_coast",
    strategy_name: str = "EqualStepAlongLineStrategy",
    source_mode: str = "all_lines",
    strategy_params: dict | None = None,
) -> MagicMock:
    meta = MagicMock()
    meta.source_dataset_name = source_dataset_name
    meta.strategy_name = strategy_name
    meta.source_mode = source_mode
    meta.strategy_params = (
        strategy_params if strategy_params is not None else {"step_m": 200.0}
    )
    return meta


def _make_point_set(
    coords: list[tuple[float, float]],
    crs: str | None = "EPSG:4326",
    **meta_kwargs,
) -> MagicMock:
    geoms = [Point(lon, lat) for lon, lat in coords]
    gdf = GeoDataFrame(geometry=geoms, crs=crs)
    ps = MagicMock()
    ps.meta = _make_meta(**meta_kwargs)
    ps.gdf = gdf
    return ps


class _ExporterTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "coastline.db"
        os.environ["COASTLINE_DATABASE_URL"] = (
            f"sqlite:///{self._db_path.as_posix()}"
        )
        import importlib
        import src.coastline.storage.db as db_module
        importlib.reload(db_module)
        import src.coastline.exporters.SQLitePointExporter as exp_module
        importlib.reload(exp_module)
        self.exporter_cls = exp_module.SQLitePointExporter

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("COASTLINE_DATABASE_URL", None)

    def _export(self, point_set) -> Path:
        return self.exporter_cls().export(point_set, output_path="")

    def _raw(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)


class TestCoordStorage(_ExporterTestBase):
    COORDS = [(37.875123, 44.125456), (37.876789, 44.126012)]

    def test_stored_coords_close_to_original(self) -> None:
        self._export(_make_point_set(self.COORDS))
        con = self._raw()
        rows = con.execute(
            "SELECT seq, lon, lat FROM coastline_points ORDER BY seq"
        ).fetchall()
        con.close()
        self.assertEqual(len(rows), len(self.COORDS))
        for idx, (exp_lon, exp_lat) in enumerate(self.COORDS):
            seq, lon, lat = rows[idx]
            self.assertEqual(seq, idx)
            self.assertAlmostEqual(lon, exp_lon, places=6)
            self.assertAlmostEqual(lat, exp_lat, places=6)


class TestPointOrder(_ExporterTestBase):
    def test_seq_is_sequential_from_zero(self) -> None:
        coords = [(37.0 + i * 0.001, 44.0) for i in range(10)]
        self._export(_make_point_set(coords))
        con = self._raw()
        seqs = [r[0] for r in con.execute(
            "SELECT seq FROM coastline_points ORDER BY seq"
        ).fetchall()]
        con.close()
        self.assertEqual(seqs, list(range(10)))


class TestIdempotency(_ExporterTestBase):
    def test_second_export_reuses_source(self) -> None:
        ps = _make_point_set([(37.875, 44.125), (37.876, 44.126)])
        self._export(ps)
        self._export(ps)
        con = self._raw()
        count = con.execute(
            "SELECT COUNT(*) FROM coastline_sources"
        ).fetchone()[0]
        con.close()
        self.assertEqual(count, 1)


class TestSourceMetadata(_ExporterTestBase):
    def test_strategy_params_stored_as_json(self) -> None:
        params = {"step_m": 150.0, "include_endpoints": True}
        self._export(_make_point_set([(37.875, 44.125)], strategy_params=params))
        con = self._raw()
        raw = con.execute(
            "SELECT strategy_params FROM coastline_sources"
        ).fetchone()[0]
        con.close()
        self.assertIsInstance(raw, str)
        self.assertEqual(json.loads(raw), params)

    def test_crs_stored(self) -> None:
        self._export(_make_point_set([(37.875, 44.125)], crs="EPSG:4326"))
        con = self._raw()
        crs = con.execute("SELECT crs FROM coastline_sources").fetchone()[0]
        con.close()
        self.assertEqual(crs, "EPSG:4326")

    def test_no_crs_stored_as_null(self) -> None:
        self._export(_make_point_set([(37.875, 44.125)], crs=None))
        con = self._raw()
        crs = con.execute("SELECT crs FROM coastline_sources").fetchone()[0]
        con.close()
        self.assertIsNone(crs)

    def test_points_count_matches(self) -> None:
        coords = [(37.0 + i * 0.001, 44.0) for i in range(7)]
        self._export(_make_point_set(coords))
        con = self._raw()
        count = con.execute(
            "SELECT points_count FROM coastline_sources"
        ).fetchone()[0]
        con.close()
        self.assertEqual(count, 7)

    def test_strategy_name_and_mode_stored(self) -> None:
        self._export(_make_point_set(
            [(37.875, 44.125)],
            strategy_name="EqualStepAlongLineStrategy",
            source_mode="main_line",
        ))
        con = self._raw()
        row = con.execute(
            "SELECT strategy_name, source_mode FROM coastline_sources"
        ).fetchone()
        con.close()
        self.assertEqual(row[0], "EqualStepAlongLineStrategy")
        self.assertEqual(row[1], "main_line")


class TestCascadeDelete(_ExporterTestBase):
    def test_delete_source_removes_points(self) -> None:
        self._export(_make_point_set([(37.875, 44.125), (37.876, 44.126)]))
        con = self._raw()
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM coastline_sources")
        con.commit()
        remaining = con.execute(
            "SELECT COUNT(*) FROM coastline_points"
        ).fetchone()[0]
        con.close()
        self.assertEqual(remaining, 0)


class TestEmptyPointSet(_ExporterTestBase):
    def test_empty_set_creates_source_but_no_points(self) -> None:
        self._export(_make_point_set([]))
        con = self._raw()
        source_count = con.execute(
            "SELECT COUNT(*) FROM coastline_sources"
        ).fetchone()[0]
        point_count = con.execute(
            "SELECT COUNT(*) FROM coastline_points"
        ).fetchone()[0]
        con.close()
        self.assertEqual(source_count, 1)
        self.assertEqual(point_count, 0)


class TestReturnedPath(_ExporterTestBase):
    def test_returns_existing_db_path(self) -> None:
        result = self._export(_make_point_set([(37.875, 44.125)]))
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        self.assertEqual(result.resolve(), self._db_path.resolve())


if __name__ == "__main__":
    unittest.main()
