from __future__ import annotations

import geopandas as gpd
from loguru import logger
from shapely.geometry import Point
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.coastline.domain.CoastlinePointSet import CoastlinePointSet, PointSetMeta
from src.coastline.storage.models import (
    COORD_SCALE,
    CoastlinePointModel,
    CoastlineSourceModel,
)


class CoastlinePointRepository:
    """
    Читает CoastlinePointSet из SQLite по id источника.

    Пример
    ------
    with SessionLocal() as session:
        repo = CoastlinePointRepository(session)
        point_set = repo.load(point_source_id=1)
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._log = logger.bind(cls=self.__class__.__name__)

    def load(self, point_source_id: int) -> CoastlinePointSet:
        """Загружает CoastlinePointSet по id из coastline_sources."""
        source = self._session.get(CoastlineSourceModel, point_source_id)
        if source is None:
            raise ValueError(
                f"CoastlineSource id={point_source_id} not found. "
                "Run point import pipeline first."
            )

        rows = (
            self._session.execute(
                select(CoastlinePointModel)
                .where(CoastlinePointModel.source_id == point_source_id)
                .order_by(CoastlinePointModel.seq)
            )
            .scalars()
            .all()
        )

        if not rows:
            raise ValueError(
                f"CoastlineSource id={point_source_id} has no points "
                "in coastline_points."
            )

        records = [
            {
                "seq": r.seq,
                "geometry": Point(r.lon_i / COORD_SCALE, r.lat_i / COORD_SCALE),
            }
            for r in rows
        ]

        crs = source.crs or "EPSG:4326"
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

        meta = PointSetMeta(
            name=source.name,
            source_dataset_name=source.name,
            strategy_name=source.strategy_name or "unknown",
            source_mode=source.source_mode or "unknown",
            points_count=len(gdf),
        )

        self._log.info(
            f"Loaded {len(gdf)} points from source id={point_source_id} "
            f"name={source.name!r} crs={crs}"
        )

        return CoastlinePointSet(gdf=gdf, meta=meta)
