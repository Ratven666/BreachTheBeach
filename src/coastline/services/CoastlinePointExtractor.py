# src/coastline/services/CoastlinePointExtractor.py
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.coastline.domain.CoastlineDataset import CoastlineDataset
from src.coastline.domain.CoastlinePointSet import CoastlinePointSet
from src.coastline.point_strategies.PointExtractionStrategy import PointExtractionStrategy


class CoastlinePointExtractor:
    """
    Application service: получает CoastlineDataset и стратегию,
    запускает извлечение точек и возвращает CoastlinePointSet.
    """

    def extract(
        self,
        dataset: CoastlineDataset,
        strategy: PointExtractionStrategy,
        name: str | None = None,
    ) -> CoastlinePointSet:
        point_set_name = name or dataset.name

        log = logger.bind(cls="CoastlinePointExtractor", name=point_set_name)
        log.info(f"Extracting points via {strategy.name}")

        records = strategy.extract(dataset)
        log.info(f"Extracted {len(records)} raw records")

        gdf = self._to_geodataframe(records, crs=dataset.crs)

        # build_meta() берёт strategy.params и собирает PointSetMeta с
        # полным набором параметров стратегии (step_m, radius_step_m и т.д.)
        meta = strategy.build_meta(
            dataset_name=dataset.name,
            point_set_name=point_set_name,
            points_count=len(gdf),
        )

        log.success(f"PointSet ready: {meta.points_count} points")
        return CoastlinePointSet(gdf=gdf, meta=meta)

    # ------------------------------------------------------------------

    @staticmethod
    def _to_geodataframe(records: list[dict], crs) -> gpd.GeoDataFrame:
        if not records:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)

        df = pd.DataFrame(records)
        return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
