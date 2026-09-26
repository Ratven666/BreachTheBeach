from __future__ import annotations

"""
src/coastline/exporters/SQLiteToGeoJsonExporter.py

Экспортирует точки береговой линии из SQLite → GeoJSON.
Читает координаты напрямую как Float (lon/lat), без COORD_SCALE.
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import Point
from sqlalchemy import select

from src.coastline.storage import db as _db


def _get_session():
    return _db.SessionLocal()


class SQLiteToGeoJsonExporter:
    """Экспортирует точки из coastline_points в GeoJSON-файл."""

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы                                             #
    # ------------------------------------------------------------------ #

    def list_sources(self) -> list[dict]:
        """Возвращает список всех источников из coastline_sources."""
        # Импортируем модели здесь, чтобы не требовать env при импорте модуля
        from coastline.storage import CoastlineSourceModel

        with _get_session() as session:

            rows = session.execute(
                select(CoastlineSourceModel).order_by(CoastlineSourceModel.id)
            ).scalars().all()

            return [
                {
                    "id":            r.id,
                    "name":          r.name,
                    "strategy_name": r.strategy_name,
                    "source_mode":   r.source_mode,
                    "points_count":  r.points_count,
                    "crs":           r.crs,
                    "created_at":    r.created_at,
                }
                for r in rows
            ]

    def _load_points(self, source_id: int, crs: str | None) -> gpd.GeoDataFrame:
        """Загружает точки источника и возвращает GeoDataFrame."""
        from coastline.storage import CoastlinePointModel

        with _get_session() as session:
            rows = session.execute(
                select(CoastlinePointModel)
                .where(CoastlinePointModel.source_id == source_id)
                .order_by(CoastlinePointModel.seq)
            ).scalars().all()

        records = [
            {
                "id":        r.id,
                "source_id": r.source_id,
                "seq":       r.seq,
                "geometry":  Point(r.lon, r.lat),
            }
            for r in rows
        ]

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs or "EPSG:4326")
        return gdf

    # ------------------------------------------------------------------ #
    #  Публичные методы экспорта                                          #
    # ------------------------------------------------------------------ #

    def export_by_id(
        self,
        source_id: int,
        output_path: Path | str,
    ) -> Path:
        """Экспортирует точки конкретного источника по его id."""
        from coastline.storage import CoastlineSourceModel

        with _get_session() as session:

            source = session.get(CoastlineSourceModel, source_id)
            if source is None:
                raise ValueError(f"CoastlineSource id={source_id} not found")
            crs  = source.crs
            name = source.name

        logger.info(f"Exporting source id={source_id} name={name!r}  crs={crs}")
        gdf = self._load_points(source_id=source_id, crs=crs)
        return self._write(gdf, output_path)

    def export(
        self,
        source_name: str,
        output_path: Path | str,
        strategy_name: str | None = None,
        source_mode: str | None = None,
    ) -> Path:
        """
        Экспортирует точки источника по имени.
        Если совпадений несколько — берёт последний (наибольший id).
        strategy_name и source_mode — опциональные уточняющие фильтры.
        """
        from coastline.storage import CoastlineSourceModel

        with _get_session() as session:
            q = (
                select(CoastlineSourceModel)
                .where(CoastlineSourceModel.name == source_name)
                .order_by(CoastlineSourceModel.id.desc())
            )
            if strategy_name is not None:
                q = q.where(CoastlineSourceModel.strategy_name == strategy_name)
            if source_mode is not None:
                q = q.where(CoastlineSourceModel.source_mode == source_mode)

            source = session.execute(q).scalars().first()
            if source is None:
                raise ValueError(
                    f"CoastlineSource name={source_name!r} not found "
                    f"(strategy={strategy_name!r}, mode={source_mode!r})"
                )
            source_id = source.id
            crs       = source.crs
            name      = source.name

        logger.info(f"Exporting source id={source_id} name={name!r}  crs={crs}")
        gdf = self._load_points(source_id=source_id, crs=crs)
        return self._write(gdf, output_path)

    # ------------------------------------------------------------------ #
    #  Запись                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write(gdf: gpd.GeoDataFrame, output_path: Path | str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out, driver="GeoJSON")
        logger.info(f"Written {len(gdf)} features → {out}")
        return out
