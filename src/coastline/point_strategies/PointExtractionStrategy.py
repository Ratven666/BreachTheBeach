from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator, Iterable

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


class PointSource(str, Enum):
    MAIN_ONLY = "main_only"
    ALL_LINES = "all_lines"


class PointExtractionStrategy(ABC):
    """
    Базовая стратегия. Подклассы реализуют только _extract_from_lines().

    Важно:
    - _iter_lines() -> атомарные LineString, с раскрытием MultiLineString;
    - _iter_merged_lines() -> линии, объединённые в непрерывные трассы,
      если это возможно.

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
        self,
        gdf: gpd.GeoDataFrame,
    ) -> Generator[tuple[str | int, LineString], None, None]:
        """
        Итерирует по всем атомарным LineString, раскрывая MultiLineString.
        Подходит для стратегий, которым важны отдельные сегменты/части.
        """
        for row_id, row in gdf.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, LineString):
                if len(geom.coords) >= 2:
                    yield row_id, geom

            elif isinstance(geom, MultiLineString):
                for part_idx, part in enumerate(geom.geoms):
                    if part is not None and not part.is_empty and len(part.coords) >= 2:
                        yield f"{row_id}_{part_idx}", part

    def _iter_feature_geometries(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> Generator[tuple[str | int, LineString | MultiLineString], None, None]:
        """
        Итерирует по исходным геометриям features без раскрытия на части.
        Удобно для стратегий, которым нужно сначала попытаться обработать
        feature как единую трассу.
        """
        for row_id, row in gdf.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, (LineString, MultiLineString)):
                yield row_id, geom

    def _iter_merged_lines(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> Generator[tuple[str | int, LineString], None, None]:
        """
        Итерирует по линиям, объединённым в непрерывные трассы там,
        где это возможно.

        Поведение:
        - LineString -> возвращается как есть;
        - MultiLineString -> сначала linemerge();
        - если после linemerge остаётся несколько частей, они возвращаются отдельно.

        Это helper для стратегий равного шага / радиуса вдоль трассы.
        """
        for row_id, geom in self._iter_feature_geometries(gdf):
            if isinstance(geom, LineString):
                if len(geom.coords) >= 2:
                    yield row_id, geom
                continue

            merged = linemerge(geom)

            if isinstance(merged, LineString):
                if len(merged.coords) >= 2:
                    yield row_id, merged

            elif isinstance(merged, MultiLineString):
                for part_idx, part in enumerate(merged.geoms):
                    if part is not None and not part.is_empty and len(part.coords) >= 2:
                        yield f"{row_id}_{part_idx}", part

    def _collect_lines(self, gdf: gpd.GeoDataFrame) -> list[LineString]:
        """
        Все атомарные линии списком.
        """
        return [line for _, line in self._iter_lines(gdf)]

    def _collect_merged_lines(self, gdf: gpd.GeoDataFrame) -> list[tuple[str | int, LineString]]:
        """
        Все объединённые линии списком.
        """
        return [(line_id, line) for line_id, line in self._iter_merged_lines(gdf)]
