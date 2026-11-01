from __future__ import annotations

"""
src/coastline/exporters/SQLiteNormalsToGeoJsonExporter.py

Экспортирует нормали береговой линии из SQLite → GeoJSON.
Геометрия — LineString от исходной точки до конца вектора нормали.
Длина отрезка задаётся параметром normal_length_m (в метрах, > 0).

Логика проекций
---------------
  1. Координаты точек (p.lon / p.lat) хранятся в БД в градусах (EPSG:4326).
  2. nx/ny — единичные векторы в метрической working_crs из БД (напр. EPSG:32637).
  3. Для корректного построения нормали:
       a) точку start берём из p.lon/p.lat → Point в EPSG:4326
       b) создаём GDF в EPSG:4326, перепроецируем в working_crs (метры)
       c) в метрической CRS считаем end = Point(x + nx*L, y + ny*L)
       d) перепроецируем [start_metric, end_metric] → LineString в EPSG:4326
  4. Итоговый GeoJSON записывается в EPSG:4326.

Пример
------
from src.coastline.exporters.SQLiteNormalsToGeoJsonExporter import (
    SQLiteNormalsToGeoJsonExporter,
)

exporter = SQLiteNormalsToGeoJsonExporter()

for ns in exporter.list_sources():
    print(ns)

exporter.export_by_id(
    normal_source_id=1,
    output_path="output/normals.geojson",
    normal_length_m=300.0,
)

exporter.export(
    source_name="nvrsk_normals_step200m_right",
    output_path="output/normals.geojson",
    normal_length_m=300.0,
)
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, Point
from sqlalchemy import select

from src.coastline.storage import db as _db

# Координаты в БД хранятся в градусах
_DB_CRS = "EPSG:4326"
# GeoJSON всегда пишется в градусах
OUTPUT_CRS = "EPSG:4326"
# Fallback метрическая СК, если working_crs не заполнена в БД
_FALLBACK_METRIC_CRS = "EPSG:32637"


def _get_session():
    return _db.SessionLocal()


class SQLiteNormalsToGeoJsonExporter:
    """Экспортирует нормали из coastline_normals в GeoJSON-файл (LineString, EPSG:4326)."""

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы                                             #
    # ------------------------------------------------------------------ #

    def list_sources(self) -> list[dict]:
        """Возвращает список всех наборов нормалей из coastline_normal_sources."""
        from src.coastline.storage.models import CoastlineNormalSourceModel

        with _get_session() as session:
            rows = (
                session.execute(
                    select(CoastlineNormalSourceModel).order_by(
                        CoastlineNormalSourceModel.id
                    )
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id":              r.id,
                    "name":            r.name,
                    "point_source_id": r.point_source_id,
                    "sea_side":        r.sea_side,
                    "tangent_delta_m": r.tangent_delta_m,
                    "working_crs":     r.working_crs,
                    "result_crs":      r.result_crs,
                    "normals_count":   r.normals_count,
                    "created_at":      r.created_at,
                }
                for r in rows
            ]

    @staticmethod
    def _validate_length(normal_length_m: float) -> None:
        if not isinstance(normal_length_m, (int, float)) or normal_length_m <= 0:
            raise ValueError(
                f"normal_length_m must be a positive number, got {normal_length_m!r}"
            )

    @staticmethod
    def _resolve_metric_crs(working_crs: str | None, source_id: int) -> str:
        """
        Возвращает метрическую CRS из записи БД (в ней хранятся nx/ny).
        Если поле пустое — логирует предупреждение и возвращает fallback.
        """
        if working_crs:
            return working_crs
        logger.warning(
            f"CoastlineNormalSource id={source_id} has no working_crs, "
            f"falling back to {_FALLBACK_METRIC_CRS}"
        )
        return _FALLBACK_METRIC_CRS

    def _load_lines(
        self,
        normal_source_id: int,
        working_crs: str | None,
        normal_length_m: float,
    ) -> gpd.GeoDataFrame:
        """
        Алгоритм:
          1. Читаем точки (lon/lat в градусах) и нормали (nx/ny в метрической CRS) из БД.
          2. Создаём GDF точек в EPSG:4326, перепроецируем в метрическую working_crs.
          3. В метрической CRS строим конец нормали: end = start + (nx, ny) * length_m.
          4. Строим LineString [start_metric, end_metric] в метрической CRS.
          5. Перепроецируем весь GDF в EPSG:4326 для записи в GeoJSON.
        """
        from src.coastline.storage.models import (
            CoastlineNormalModel,
            CoastlinePointModel,
        )

        with _get_session() as session:
            rows = session.execute(
                select(CoastlineNormalModel, CoastlinePointModel)
                .join(
                    CoastlinePointModel,
                    CoastlineNormalModel.point_id == CoastlinePointModel.id,
                )
                .where(CoastlineNormalModel.normal_source_id == normal_source_id)
                .order_by(CoastlineNormalModel.id)
            ).all()

        # Метрическая CRS из БД — в ней хранятся nx/ny
        metric_crs = self._resolve_metric_crs(working_crs, normal_source_id)

        # Шаг 1: точки в градусах → GDF в EPSG:4326
        start_points = []
        attrs_list = []
        for n, p in rows:
            start_points.append(Point(p.lon, p.lat))
            attrs_list.append(
                {
                    "normal_id":          n.id,
                    "normal_source_id":   n.normal_source_id,
                    "point_id":           n.point_id,
                    "nx":                 n.nx,
                    "ny":                 n.ny,
                    "normal_azimuth_deg": n.normal_azimuth_deg,
                    "normal_length_m":    normal_length_m,
                }
            )

        gdf_starts = gpd.GeoDataFrame(
            attrs_list,
            geometry=start_points,
            crs=_DB_CRS,
        )

        # Шаг 2: перепроецируем стартовые точки в метрическую CRS
        gdf_metric = gdf_starts.to_crs(metric_crs)

        # Шаг 3 & 4: строим LineString в метрической CRS
        lines = []
        for _, row in gdf_metric.iterrows():
            sx = row.geometry.x
            sy = row.geometry.y
            ex = sx + row["nx"] * normal_length_m
            ey = sy + row["ny"] * normal_length_m
            lines.append(LineString([(sx, sy), (ex, ey)]))

        gdf_metric = gdf_metric.copy()
        gdf_metric["geometry"] = lines
        gdf_metric = gdf_metric.set_geometry("geometry")

        # Шаг 5: перепроецируем готовые линии в EPSG:4326
        return gdf_metric.to_crs(OUTPUT_CRS)

    @staticmethod
    def _write(gdf: gpd.GeoDataFrame, output_path: Path | str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out, driver="GeoJSON")
        logger.info(f"Written {len(gdf)} features ({gdf.crs}) → {out}")
        return out

    # ------------------------------------------------------------------ #
    #  Публичные методы экспорта                                          #
    # ------------------------------------------------------------------ #

    def export_by_id(
        self,
        normal_source_id: int,
        output_path: Path | str,
        normal_length_m: float,
    ) -> Path:
        """Экспортирует нормали как LineString по id набора."""
        self._validate_length(normal_length_m)

        from src.coastline.storage.models import CoastlineNormalSourceModel

        with _get_session() as session:
            ns = session.get(CoastlineNormalSourceModel, normal_source_id)
            if ns is None:
                raise ValueError(
                    f"CoastlineNormalSource id={normal_source_id} not found"
                )
            working_crs = ns.working_crs
            name = ns.name

        logger.info(
            f"Exporting normals id={normal_source_id} name={name!r} "
            f"db_crs={_DB_CRS} metric_crs={working_crs!r} "
            f"length={normal_length_m} m → {OUTPUT_CRS}"
        )
        gdf = self._load_lines(normal_source_id, working_crs, normal_length_m)
        return self._write(gdf, output_path)

    def export(
        self,
        source_name: str,
        output_path: Path | str,
        normal_length_m: float,
        sea_side: str | None = None,
    ) -> Path:
        """
        Экспортирует нормали как LineString по имени набора.
        Если совпадений несколько — берёт последний (наибольший id).
        sea_side — опциональный уточняющий фильтр.
        """
        self._validate_length(normal_length_m)

        from src.coastline.storage.models import CoastlineNormalSourceModel

        with _get_session() as session:
            q = (
                select(CoastlineNormalSourceModel)
                .where(CoastlineNormalSourceModel.name == source_name)
                .order_by(CoastlineNormalSourceModel.id.desc())
            )
            if sea_side is not None:
                q = q.where(CoastlineNormalSourceModel.sea_side == sea_side)

            ns = session.execute(q).scalars().first()
            if ns is None:
                raise ValueError(
                    f"CoastlineNormalSource name={source_name!r} not found"
                    + (f" (sea_side={sea_side!r})" if sea_side else "")
                )
            normal_source_id = ns.id
            working_crs = ns.working_crs
            name = ns.name

        logger.info(
            f"Exporting normals id={normal_source_id} name={name!r} "
            f"db_crs={_DB_CRS} metric_crs={working_crs!r} "
            f"length={normal_length_m} m → {OUTPUT_CRS}"
        )
        gdf = self._load_lines(normal_source_id, working_crs, normal_length_m)
        return self._write(gdf, output_path)