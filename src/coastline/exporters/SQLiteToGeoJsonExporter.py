from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import Point
from sqlalchemy import select

from src.coastline.storage import db as _db_module
from src.coastline.storage.models import (
    Base,
    CoastlinePointModel,
    CoastlineSourceModel,
    COORD_SCALE,
)


class SQLiteToGeoJsonExporter:
    """
    Читает точки береговой линии из SQLite-базы и сохраняет в GeoJSON.

    Использование
    -------------
    exporter = SQLiteToGeoJsonExporter()

    # По имени источника (берёт последнюю запись, если их несколько)
    path = exporter.export(
        source_name="nvrsk_main_coastline",
        output_path="out/points.geojson",
    )

    # Уточнение по стратегии и режиму
    path = exporter.export(
        source_name="nvrsk_main_coastline",
        output_path="out/points.geojson",
        strategy_name="EqualStepAlongLineStrategy",
        source_mode="all_lines",
    )

    # По первичному ключу источника
    path = exporter.export_by_id(source_id=1, output_path="out/points.geojson")

    # Список всех источников в БД
    sources = exporter.list_sources()
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        source_name: str,
        output_path: str | Path,
        strategy_name: str | None = None,
        source_mode: str | None = None,
    ) -> Path:
        """
        Экспортирует точки источника с заданным именем в GeoJSON.

        Если в БД несколько записей с одним именем, можно уточнить
        через ``strategy_name`` и ``source_mode``.
        При нескольких совпадениях берётся самая новая (по id DESC).
        """
        meta, coords, crs = self._load(source_name, strategy_name, source_mode)
        return self._write(meta, coords, crs, Path(output_path))

    def export_by_id(self, source_id: int, output_path: str | Path) -> Path:
        """Экспортирует точки конкретного источника по его первичному ключу."""
        self._ensure_schema()
        with _db_module.SessionLocal() as session:
            source = session.get(CoastlineSourceModel, source_id)
            if source is None:
                raise ValueError(f"CoastlineSource id={source_id} not found")
            meta, coords, crs = self._read_source(source, session)
        return self._write(meta, coords, crs, Path(output_path))

    def list_sources(self) -> list[dict]:
        """Возвращает список всех источников в БД (без точек)."""
        self._ensure_schema()
        with _db_module.SessionLocal() as session:
            rows = session.execute(
                select(CoastlineSourceModel).order_by(CoastlineSourceModel.id)
            ).scalars().all()
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "strategy_name": r.strategy_name,
                    "source_mode": r.source_mode,
                    "crs": r.crs,
                    "points_count": r.points_count,
                    "created_at": str(r.created_at),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        Base.metadata.create_all(_db_module.engine)

    def _load(
        self,
        name: str,
        strategy_name: str | None,
        source_mode: str | None,
    ) -> tuple[dict, list[tuple[float, float]], str | None]:
        self._ensure_schema()
        with _db_module.SessionLocal() as session:
            stmt = select(CoastlineSourceModel).where(
                CoastlineSourceModel.name == name
            )
            if strategy_name is not None:
                stmt = stmt.where(
                    CoastlineSourceModel.strategy_name == strategy_name
                )
            if source_mode is not None:
                stmt = stmt.where(
                    CoastlineSourceModel.source_mode == source_mode
                )
            stmt = stmt.order_by(CoastlineSourceModel.id.desc())

            source = session.execute(stmt).scalars().first()
            if source is None:
                raise ValueError(
                    f"CoastlineSource not found: name={name!r}, "
                    f"strategy_name={strategy_name!r}, "
                    f"source_mode={source_mode!r}"
                )
            return self._read_source(source, session)

    @staticmethod
    def _read_source(
        source: CoastlineSourceModel,
        session,
    ) -> tuple[dict, list[tuple[float, float]], str | None]:
        """Читает метаданные и координаты из открытой сессии."""
        rows = session.execute(
            select(CoastlinePointModel)
            .where(CoastlinePointModel.source_id == source.id)
            .order_by(CoastlinePointModel.seq)
        ).scalars().all()

        meta = {
            "id": source.id,
            "name": source.name,
            "strategy_name": source.strategy_name,
            "source_mode": source.source_mode,
            "strategy_params": (
                json.loads(source.strategy_params)
                if source.strategy_params else None
            ),
            "points_count": source.points_count,
        }
        coords = [(p.lon_i / COORD_SCALE, p.lat_i / COORD_SCALE) for p in rows]
        return meta, coords, source.crs

    @staticmethod
    def _write(
        meta: dict,
        coords: list[tuple[float, float]],
        crs: str | None,
        output_path: Path,
    ) -> Path:
        output_path = output_path.with_suffix(".geojson")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        geoms = [Point(lon, lat) for lon, lat in coords]
        gdf = gpd.GeoDataFrame(
            {
                "seq": list(range(len(geoms))),
                "source_name": meta["name"],
                "strategy_name": meta["strategy_name"],
                "source_mode": meta["source_mode"],
            },
            geometry=geoms,
            crs=crs or "EPSG:4326",
        )

        gdf.to_file(output_path, driver="GeoJSON")

        logger.success(
            f"Exported {len(geoms)} points "
            f"from '{meta['name']}' (source id={meta['id']}) → {output_path}"
        )
        return output_path
