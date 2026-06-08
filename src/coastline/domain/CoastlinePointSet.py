from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from loguru import logger


@dataclass
class PointSetMeta:
    name: str
    source_dataset_name: str
    strategy_name: str
    source_mode: str
    points_count: int


class CoastlinePointSet:
    """
    Результат извлечения точек из CoastlineDataset.

    Является самостоятельным объектом: хранит точки,
    метаданные и умеет сохранять себя через PointExportStrategy.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        meta: PointSetMeta,
    ) -> None:
        self.gdf = gdf
        self.meta = meta
        self._log = logger.bind(cls="CoastlinePointSet", name=meta.name)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        strategy: "PointExportStrategy",
        output_path: str | Path,
    ) -> Path:
        from src.coastline.exporters.PointExportStrategy import PointExportStrategy  # noqa: F401

        self._log.info(
            f"Exporting {self.meta.points_count} points "
            f"via {strategy.__class__.__name__} → {output_path}"
        )
        return strategy.export(self, output_path)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        m = self.meta
        print(f"=== PointSet: {m.name} ===")
        print(f"  Dataset  : {m.source_dataset_name}")
        print(f"  Strategy : {m.strategy_name}")
        print(f"  Source   : {m.source_mode}")
        print(f"  Points   : {m.points_count}")

    def __len__(self) -> int:
        return len(self.gdf)

    def __repr__(self) -> str:
        return (
            f"CoastlinePointSet(name={self.meta.name!r}, "
            f"points={self.meta.points_count}, "
            f"strategy={self.meta.strategy_name!r})"
        )
