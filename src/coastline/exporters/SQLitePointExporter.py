# src/coastline/exporters/SQLitePointExporter.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.coastline.exporters.PointExportStrategy import PointExportStrategy
from src.coastline.storage.db import DATABASE_URL, init_db, session_scope
from src.coastline.storage.models import (
    COORD_SCALE,
    CoastlinePointModel,
    CoastlineSourceModel,
)


class SQLitePointExporter(PointExportStrategy):

    def export(self, point_set: "CoastlinePointSet", output_path: str | Path) -> Path:
        init_db()

        meta = point_set.meta
        crs: str | None = (
            point_set.gdf.crs.to_string() if point_set.gdf.crs is not None else None
        )

        raw_params = getattr(meta, "strategy_params", {})
        strategy_params_json = (
            json.dumps(raw_params, ensure_ascii=False)
            if isinstance(raw_params, dict)
            else str(raw_params)
        )

        with session_scope() as session:
            source = session.execute(
                select(CoastlineSourceModel).where(
                    CoastlineSourceModel.name == meta.source_dataset_name,
                    CoastlineSourceModel.strategy_name == meta.strategy_name,
                    CoastlineSourceModel.source_mode == meta.source_mode,
                )
            ).scalar_one_or_none()

            if source is None:
                source = CoastlineSourceModel(
                    name=meta.source_dataset_name,
                    geojson_path=str(getattr(meta, "geojson_path", "")),
                    strategy_name=meta.strategy_name,
                    source_mode=meta.source_mode,
                    strategy_params=strategy_params_json,
                    crs=crs,
                    points_count=len(point_set.gdf),
                    created_at=datetime.utcnow(),
                )
                session.add(source)
                try:
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    source = session.execute(
                        select(CoastlineSourceModel).where(
                            CoastlineSourceModel.name == meta.source_dataset_name,
                            CoastlineSourceModel.strategy_name == meta.strategy_name,
                            CoastlineSourceModel.source_mode == meta.source_mode,
                        )
                    ).scalar_one()

            points = [
                CoastlinePointModel(
                    source_id=source.id,
                    seq=idx,
                    lon_i=round(geom.x * COORD_SCALE),
                    lat_i=round(geom.y * COORD_SCALE),
                )
                for idx, geom in enumerate(point_set.gdf.geometry)
            ]
            session.bulk_save_objects(points)

        logger.info(
            f"Point export → {DATABASE_URL} | "
            f"source={meta.source_dataset_name!r} "
            f"strategy={meta.strategy_name!r} "
            f"({len(points)} pts)"
        )

        if DATABASE_URL.startswith("sqlite:///"):
            return Path(DATABASE_URL.removeprefix("sqlite:///"))
        return Path(".")
