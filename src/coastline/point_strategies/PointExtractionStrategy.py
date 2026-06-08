from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString


class PointSource(str, Enum):
    MAIN_ONLY = "main_only"
    ALL_LINES = "all_lines"


class PointExtractionStrategy(ABC):
    """
    Базовая стратегия. Подклассы реализуют только _extract_from_lines().
    Выбор источника (main / all) вынесен сюда.
    """

    def __init__(self, source: PointSource = PointSource.ALL_LINES) -> None:
        self.source = source

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, dataset: "CoastlineDataset") -> list[dict]:
        """
        Вернуть список dict-записей; каждая запись содержит 'geometry'
        и произвольные атрибуты. Список передаётся в CoastlinePointExtractor.
        """
        source_gdf = self._pick_source(dataset)
        return self._extract_from_lines(source_gdf)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_from_lines(self, gdf: gpd.GeoDataFrame) -> list[dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _pick_source(self, dataset: "CoastlineDataset") -> gpd.GeoDataFrame:
        if self.source == PointSource.MAIN_ONLY:
            return dataset.main_gdf.copy()
        return dataset.combined_gdf.copy()

    def _iter_lines(
        self, gdf: gpd.GeoDataFrame
    ) -> Generator[tuple[str | int, LineString], None, None]:
        """Итерирует по всем LineString, раскрывая MultiLineString."""
        for row_id, row in gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString) and not geom.is_empty:
                yield row_id, geom
            elif isinstance(geom, MultiLineString):
                for part_idx, part in enumerate(geom.geoms):
                    if not part.is_empty:
                        yield f"{row_id}_{part_idx}", part
