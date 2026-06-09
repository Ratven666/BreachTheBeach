from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import box


# ============================================================
# Value object: BBox
# ============================================================

@dataclass(frozen=True, slots=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError("latitude must be in [-90, 90]")
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("longitude must be in [-180, 180]")

    def to_overpass_bbox(self) -> str:
        # Overpass: south, west, north, east
        return f"{self.south},{self.west},{self.north},{self.east}"

    def to_osmium_bbox(self) -> str:
        # Osmium: min_lon,min_lat,max_lon,max_lat
        return f"{self.west},{self.south},{self.east},{self.north}"

    def to_tuple_wsen(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north

    def to_polygon(self):
        return box(self.west, self.south, self.east, self.north)


# ============================================================
# Base abstract extractor
# ============================================================

class BBoxExtractor(ABC):
    """
    Единый базовый абстрактный extractor.

    Контракт:
    - fetch() получает сырые данные
    - parse() преобразует их в GeoDataFrame
    - extract() запускает общий pipeline
    - итог всегда GeoDataFrame в EPSG:4326,
      строго обрезанный по bbox
    """

    def __init__(self, bbox: BBox, output_path: str | Path | None = None) -> None:
        self.bbox = bbox
        self.output_path = Path(output_path) if output_path else None

    def extract(self) -> gpd.GeoDataFrame:
        self.validate()
        raw_data = self.fetch()
        gdf = self.parse(raw_data)
        gdf = self.postprocess(gdf)

        if self.output_path is not None:
            self.save(gdf, self.output_path)

        return gdf

    def validate(self) -> None:
        _ = self.bbox.to_tuple_wsen()

    @abstractmethod
    def fetch(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw_data: Any) -> gpd.GeoDataFrame:
        raise NotImplementedError

    def postprocess(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None:
            raise ValueError("parse() returned None instead of GeoDataFrame")

        if gdf.empty:
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)
            return gdf.reset_index(drop=True)

        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

        bbox_mask = gpd.GeoDataFrame(
            {"geometry": [self.bbox.to_polygon()]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        # Точная геометрическая обрезка
        gdf = gpd.clip(gdf, bbox_mask)

        # Удаляем пустые и null-геометрии
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        return gdf.reset_index(drop=True)

    def save(self, gdf: gpd.GeoDataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix in {".geojson", ".json"}:
            gdf.to_file(output_path, driver="GeoJSON")
        elif suffix == ".gpkg":
            gdf.to_file(output_path, driver="GPKG")
        elif suffix == ".shp":
            gdf.to_file(output_path, driver="ESRI Shapefile")
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
