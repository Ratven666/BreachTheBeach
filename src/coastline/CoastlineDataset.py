from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.coastline.exporters.CoastlineExportStrategy import CoastlineExportStrategy


@dataclass
class CoastlineSummary:
    crs: str | None
    actual_bbox: tuple[float, float, float, float]
    main_feature_count: int
    other_feature_count: int
    total_feature_count: int
    main_total_length: float | None
    other_total_length: float | None
    total_length: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "actual_bbox": self.actual_bbox,
            "main_feature_count": self.main_feature_count,
            "other_feature_count": self.other_feature_count,
            "total_feature_count": self.total_feature_count,
            "main_total_length": self.main_total_length,
            "other_total_length": self.other_total_length,
            "total_length": self.total_length,
        }

class CoastlineDataset:
    """
    Единый объект для работы с основной и дополнительной береговыми линиями.

    Возможности:
    - чтение из двух GeoJSON;
    - вычисление фактического bbox;
    - текстовая/словарная сводка;
    - экспорт через паттерн Strategy.
    """

    def __init__(
        self,
        main_gdf: gpd.GeoDataFrame,
        other_gdf: gpd.GeoDataFrame,
        name: str = "coastline_dataset",
    ) -> None:
        self.log = logger.bind(dataset=self.__class__.__name__, name=name)
        self.name = name

        self.main_gdf = self._prepare_gdf(main_gdf, role="main_coastline")
        self.other_gdf = self._prepare_gdf(other_gdf, role="other_coastline")

        self._align_crs()

        self.log.debug(
            f"Dataset initialized: main={len(self.main_gdf)}, other={len(self.other_gdf)}"
        )

    # ------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------

    @classmethod
    def from_geojson(
        cls,
        main_path: str | Path,
        other_path: str | Path,
        name: str = "coastline_dataset",
    ) -> "CoastlineDataset":
        main_path = Path(main_path)
        other_path = Path(other_path)

        log = logger.bind(dataset=cls.__name__, name=name)
        log.info(f"Loading main coastline GeoJSON: {main_path}")
        log.info(f"Loading other coastline GeoJSON: {other_path}")

        main_gdf = gpd.read_file(main_path)
        other_gdf = gpd.read_file(other_path)

        return cls(main_gdf=main_gdf, other_gdf=other_gdf, name=name)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _prepare_gdf(self, gdf: gpd.GeoDataFrame, role: str) -> gpd.GeoDataFrame:
        if gdf is None:
            raise ValueError(f"{role}: GeoDataFrame is None")

        if "geometry" not in gdf.columns:
            raise ValueError(f"{role}: missing geometry column")

        gdf = gdf.copy()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

        if "coastline_role" not in gdf.columns:
            gdf["coastline_role"] = role

        return gdf

    def _align_crs(self) -> None:
        main_crs = self.main_gdf.crs
        other_crs = self.other_gdf.crs

        if main_crs is None and other_crs is None:
            self.log.warning("Both GeoDataFrames have no CRS, assuming EPSG:4326")
            self.main_gdf = self.main_gdf.set_crs("EPSG:4326")
            self.other_gdf = self.other_gdf.set_crs("EPSG:4326")
            return

        if main_crs is None and other_crs is not None:
            self.log.warning(f"Main CRS missing, assigning CRS from other: {other_crs}")
            self.main_gdf = self.main_gdf.set_crs(other_crs)
            return

        if other_crs is None and main_crs is not None:
            self.log.warning(f"Other CRS missing, assigning CRS from main: {main_crs}")
            self.other_gdf = self.other_gdf.set_crs(main_crs)
            return

        if main_crs != other_crs:
            self.log.info(f"Reprojecting other coastline from {other_crs} to {main_crs}")
            self.other_gdf = self.other_gdf.to_crs(main_crs)

    # ------------------------------------------------------------
    # Public info API
    # ------------------------------------------------------------

    @property
    def crs(self):
        return self.main_gdf.crs if self.main_gdf.crs is not None else self.other_gdf.crs

    @property
    def combined_gdf(self) -> gpd.GeoDataFrame:
        combined = pd.concat([self.main_gdf, self.other_gdf], ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=self.crs)

    @property
    def actual_bbox(self) -> tuple[float, float, float, float]:
        """
        Возвращает фактические границы всех линий:
        (minx, miny, maxx, maxy)
        """
        total_bounds = self.combined_gdf.total_bounds
        return (
            float(total_bounds[0]),
            float(total_bounds[1]),
            float(total_bounds[2]),
            float(total_bounds[3]),
        )

    def summary(self) -> CoastlineSummary:
        main_total_length = self._safe_total_length(self.main_gdf)
        other_total_length = self._safe_total_length(self.other_gdf)

        summary = CoastlineSummary(
            crs=str(self.crs) if self.crs is not None else None,
            actual_bbox=self.actual_bbox,
            main_feature_count=len(self.main_gdf),
            other_feature_count=len(self.other_gdf),
            total_feature_count=len(self.main_gdf) + len(self.other_gdf),
            main_total_length=main_total_length,
            other_total_length=other_total_length,
            total_length=(
                (main_total_length or 0.0) + (other_total_length or 0.0)
            ),
        )

        self.log.info(f"Summary computed: {summary.as_dict()}")
        return summary

    def summary_dict(self) -> dict[str, Any]:
        return self.summary().as_dict()

    def print_summary(self) -> None:
        summary = self.summary()

        print("Coastline dataset summary")
        print(f"CRS: {summary.crs}")
        print(
            "Actual bbox: "
            f"({summary.actual_bbox[0]}, {summary.actual_bbox[1]}) - "
            f"({summary.actual_bbox[2]}, {summary.actual_bbox[3]})"
        )
        print(f"Main features: {summary.main_feature_count}")
        print(f"Other features: {summary.other_feature_count}")
        print(f"Total features: {summary.total_feature_count}")
        print(f"Main total length: {summary.main_total_length}")
        print(f"Other total length: {summary.other_total_length}")
        print(f"Total length: {summary.total_length}")

    def _safe_total_length(self, gdf: gpd.GeoDataFrame) -> float | None:
        if gdf.empty:
            return 0.0
        try:
            return float(gdf.geometry.length.sum())
        except Exception:
            self.log.exception("Failed to calculate geometry lengths")
            return None

    # ------------------------------------------------------------
    # Export API
    # ------------------------------------------------------------

    def export(
        self,
        strategy: CoastlineExportStrategy,
        output_path: str | Path,
    ) -> Path:
        self.log.info(
            f"Exporting dataset with strategy={strategy.__class__.__name__} "
            f"to {output_path}"
        )
        return strategy.export(self, output_path)

if __name__ == "__main__":
    from loguru import logger

    from src.coastline.exporters.GeoJsonCoastlineExporter import GeoJsonCoastlineExporter
    from src.coastline.exporters.GeoPackageCoastlineExporter import GeoPackageCoastlineExporter

    logger.info("Loading coastline dataset")

    dataset = CoastlineDataset.from_geojson(
        main_path="../../output/main_coastline.geojson",
        other_path="../../output/other_lines.geojson",
        name="novoross_coastline",
    )

    dataset.print_summary()

    summary = dataset.summary_dict()
    print("\nSummary as dict:")
    print(summary)

    logger.info("Exporting to GeoJSON")
    dataset.export(
        strategy=GeoJsonCoastlineExporter(),
        output_path="../../data/exports/novoross_coastline.geojson",
    )

    logger.info("Exporting to GeoPackage")
    dataset.export(
        strategy=GeoPackageCoastlineExporter(),
        output_path="../../data/exports/novoross_coastline.gpkg",
    )

    logger.success("All exports completed")
