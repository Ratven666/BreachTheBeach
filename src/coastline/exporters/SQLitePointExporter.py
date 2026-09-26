from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from src.coastline.exporters.PointExportStrategy import PointExportStrategy
from src.coastline.storage import db as _db_module
from src.coastline.storage.models import (
    Base,
    CoastlinePointModel,
    CoastlineSourceModel,
    COORD_SCALE,
)


class SQLitePointExporter(PointExportStrategy):

    def export(self, point_set, output_path: str | Path) -> Path:
        engine = _db_module.engine
        Base.metadata.create_all(engine)

        meta = point_set.meta
        gdf = point_set.gdf

        crs_str: str | None = str(gdf.crs) if gdf.crs is not None else None

        # strategy_params и geojson_path — опциональные атрибуты,
        # которых нет в реальном PointSetMeta, но могут быть в тестах
        raw_params = getattr(meta, "strategy_params", None)
        strategy_params_json: str | None = (
            json.dumps(raw_params) if raw_params is not None else None
        )
        geojson_path: str = getattr(meta, "geojson_path", "") or ""

        Session = _db_module.SessionLocal

        with Session() as session:
            stmt = select(CoastlineSourceModel).where(
                CoastlineSourceModel.name == meta.source_dataset_name,
                CoastlineSourceModel.strategy_name == meta.strategy_name,
                CoastlineSourceModel.source_mode == meta.source_mode,
            )
            source = session.execute(stmt).scalar_one_or_none()

            if source is None:
                source = CoastlineSourceModel(
                    name=meta.source_dataset_name,
                    geojson_path=geojson_path,
                    strategy_name=meta.strategy_name,
                    source_mode=meta.source_mode,
                    strategy_params=strategy_params_json,
                    crs=crs_str,
                    points_count=len(gdf),
                )
                session.add(source)
                session.flush()

                points = [
                    CoastlinePointModel(
                        source_id=source.id,
                        seq=seq,
                        lon_i=round(geom.x * COORD_SCALE),
                        lat_i=round(geom.y * COORD_SCALE),
                    )
                    for seq, geom in enumerate(gdf.geometry)
                ]
                if points:
                    session.add_all(points)

            session.commit()

        db_url = engine.url.render_as_string(hide_password=False)
        db_path = Path(db_url.replace("sqlite:///", "", 1))
        return db_path