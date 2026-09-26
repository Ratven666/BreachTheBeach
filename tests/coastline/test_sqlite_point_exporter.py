"""
tests/coastline/test_sqlite_point_exporter.py

Тесты экспорта точек береговой линии в SQLite-базу через
SQLitePointExporter.

Проверяемые свойства:
- Координаты кодируются и декодируются без потери точности.
- Порядок (seq) точек сохраняется.
- Повторный экспорт того же источника не дублирует coastline_sources
  (idempotency).
- Метаданные источника (strategy_params как JSON-строка, crs,
  points_count) записываются корректно.
- Каскадное удаление: при удалении CoastlineSourceModel все
  связанные CoastlinePointModel удаляются.
- Пустой PointSet создаёт запись источника, но не создаёт точек.
- Возвращаемый Path указывает на реальный файл БД.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from geopandas import GeoDataFrame
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Вспомогательные фабрики — имитируют объекты domain-слоя
# ---------------------------------------------------------------------------

def _make_meta(
    name: str = "test_coast",
    strategy_name: str = "EqualStepAlongLineStrategy",
    source_mode: str = "all_lines",
    strategy_params: dict | None = None,
    geojson_path: str = "data/coastline/test.geojson",
) -> MagicMock:
    meta = MagicMock()
    meta.source_dataset_name = name
    meta.strategy_name = strategy_name
    meta.source_mode = source_mode
    meta.strategy_params = strategy_params or {"step_m": 200.0}
    meta.geojson_path = geojson_path
    return meta


def _make_point_set(
    coords: list[tuple[float, float]],
    crs: str | None = "EPSG:4326",
    **meta_kwargs: Any,
) -> MagicMock:
    """
    Создаёт MagicMock, имитирующий CoastlinePointSet.

    coords — список (lon, lat) пар в градусах WGS-84.
    """
    geoms = [Point(lon, lat) for lon, lat in coords]
    gdf = GeoDataFrame(geometry=geoms, crs=crs)

    ps = MagicMock()
    ps.meta = _make_meta(**meta_kwargs)
    ps.gdf = gdf
    return ps


# ---------------------------------------------------------------------------
# Базовый класс: изолированная БД в temp-каталоге
# ---------------------------------------------------------------------------

class _ExporterTestBase(unittest.TestCase):
    """
    Перед каждым тестом выставляет COASTLINE_DATABASE_URL на новый
    temp-файл и перезагружает db-модуль, чтобы DATABASE_URL
    пересчиталось.
    """

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

        from src.coastline.storage.models import COORD_SCALE
        self.COORD_SCALE = COORD_SCALE

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("COASTLINE_DATABASE_URL", None)

    def _export(self, point_set: MagicMock) -> Path:
        return self.exporter_cls().export(point_set, output_path="")

    def _raw(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)


# ---------------------------------------------------------------------------
# Тест 1: кодирование координат
# ---------------------------------------------------------------------------

class TestCoordEncoding(_ExporterTestBase):
    """lon_i = round(lon × COORD_SCALE), lat_i = round(lat × COORD_SCALE)."""

    COORDS = [(37.875, 44.125), (37.876, 44.126)]

    def test_encoded_values_match(self) -> None:
        self._export(_make_point_set(self.COORDS))

        con = self._raw()
        rows = con.execute(
            "SELECT seq, lon_i, lat_i FROM coastline_points ORDER BY seq"
        ).fetchall()
        con.close()

        self.assertEqual(len(rows), len(self.COORDS))
        for idx, (lon, lat) in enumerate(self.COORDS):
            seq, lon_i, lat_i = rows[idx]
            self.assertEqual(seq, idx)
            self.assertEqual(lon_i, round(lon * self.COORD_SCALE))
            self.assertEqual(lat_i, round(lat * self.COORD_SCALE))

    def test_decoded_coords_close_to_original(self) -> None:
        """Обратное декодирование: погрешность < 1e-6 °."""
        self._export(_make_point_set(self.COORDS))

        con = self._raw()
        rows = con.execute(
            "SELECT lon_i, lat_i FROM coastline_points ORDER BY seq"
        ).fetchall()
        con.close()

        for (lon_i, lat_i), (exp_lon, exp_lat) in zip(rows, self.COORDS):
            self.assertAlmostEqual(lon_i / self.COORD_SCALE, exp_lon, places=6)
            self.assertAlmostEqual(lat_i / self.COORD_SCALE, exp_lat, places=6)


# ---------------------------------------------------------------------------
# Тест 2: порядок точек
# ---------------------------------------------------------------------------

class TestPointOrder(_ExporterTestBase):
    def test_seq_is_sequential_from_zero(self) -> None:
        coords = [(37.0 + i * 0.001, 44.0) for i in range(10)]
        self._export(_make_point_set(coords))

        con = self._raw()
        seqs = [
            r[0] for r in con.execute(
                "SELECT seq FROM coastline_points ORDER BY seq"
            ).fetchall()
        ]
        con.close()

        self.assertEqual(seqs, list(range(10)))


# ---------------------------------------------------------------------------
# Тест 3: idempotency — повторный экспорт не дублирует источник
# ---------------------------------------------------------------------------

class TestIdempotency(_ExporterTestBase):
    def test_second_export_reuses_source(self) -> None:
        ps = _make_point_set([(37.875, 44.125), (37.876, 44.126)])

        self._export(ps)
        self._export(ps)

        con = self._raw()
        source_count = con.execute(
            "SELECT COUNT(*) FROM coastline_sources"
        ).fetchone()[0]
        con.close()

        self.assertEqual(source_count, 1)


# ---------------------------------------------------------------------------
# Тест 4: метаданные источника
# ---------------------------------------------------------------------------

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

    def test_points_count_matches(self) -> None:
        coords = [(37.0 + i * 0.001, 44.0) for i in range(7)]
        self._export(_make_point_set(coords))

        con = self._raw()
        count = con.execute(
            "SELECT points_count FROM coastline_sources"
        ).fetchone()[0]
        con.close()

        self.assertEqual(count, 7)

    def test_no_crs_stored_as_null(self) -> None:
        self._export(_make_point_set([(37.875, 44.125)], crs=None))

        con = self._raw()
        crs = con.execute("SELECT crs FROM coastline_sources").fetchone()[0]
        con.close()

        self.assertIsNone(crs)


# ---------------------------------------------------------------------------
# Тест 5: каскадное удаление
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Тест 6: пустой PointSet
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Тест 7: возвращаемый путь
# ---------------------------------------------------------------------------

class TestReturnedPath(_ExporterTestBase):
    def test_returns_existing_db_path(self) -> None:
        result = self._export(_make_point_set([(37.875, 44.125)]))

        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        self.assertEqual(result.resolve(), self._db_path.resolve())


if __name__ == "__main__":
    unittest.main()