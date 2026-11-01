from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.coastline.domain.CoastlineDataset import CoastlineDataset


@dataclass(frozen=True)
class CoastlineSimplificationSummary:
    dataset_name: str
    tolerance_m: float
    preserve_topology: bool
    source_crs: str | None
    metric_crs: str
    main_features_before: int
    other_features_before: int
    main_features_after: int
    other_features_after: int
    main_vertices_before: int
    other_vertices_before: int
    main_vertices_after: int
    other_vertices_after: int

    def as_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "tolerance_m": self.tolerance_m,
            "preserve_topology": self.preserve_topology,
            "source_crs": self.source_crs,
            "metric_crs": self.metric_crs,
            "main_features_before": self.main_features_before,
            "other_features_before": self.other_features_before,
            "main_features_after": self.main_features_after,
            "other_features_after": self.other_features_after,
            "main_vertices_before": self.main_vertices_before,
            "other_vertices_before": self.other_vertices_before,
            "main_vertices_after": self.main_vertices_after,
            "other_vertices_after": self.other_vertices_after,
        }


class CoastlineSimplifier:
    """
    Сервис упрощения береговой линии.

    Основные правила:
    - упрощение выполняется в метрической CRS;
    - tolerance задаётся в метрах;
    - после упрощения геометрии возвращаются в исходную CRS датасета;
    - пустые / схлопнувшиеся геометрии отфильтровываются.

    Подходит для ускорения последующих spatial-операций:
    трассировки, пересечений, spatial index query и т.д.
    """

    def __init__(self) -> None:
        self._log = logger.bind(cls="CoastlineSimplifier")

    def simplify(
        self,
        dataset: CoastlineDataset,
        tolerance_m: float,
        preserve_topology: bool = True,
        name: str | None = None,
    ) -> CoastlineDataset:
        if tolerance_m <= 0:
            raise ValueError(f"tolerance_m must be > 0, got {tolerance_m}")

        source_crs = dataset.crs
        metric_crs = dataset.metric_crs

        self._log.info(
            f"Simplifying coastline dataset '{dataset.name}' "
            f"with tolerance={tolerance_m} m, preserve_topology={preserve_topology}, "
            f"metric_crs={metric_crs}"
        )

        main_before = dataset.main_gdf.copy()
        other_before = dataset.other_gdf.copy()

        main_metric = self._to_metric(main_before, metric_crs)
        other_metric = self._to_metric(other_before, metric_crs)

        main_after_metric = self._simplify_gdf(
            main_metric,
            tolerance_m=tolerance_m,
            preserve_topology=preserve_topology,
        )
        other_after_metric = self._simplify_gdf(
            other_metric,
            tolerance_m=tolerance_m,
            preserve_topology=preserve_topology,
        )

        if source_crs is not None and str(metric_crs) != str(source_crs):
            main_after = main_after_metric.to_crs(source_crs)
            other_after = other_after_metric.to_crs(source_crs)
        else:
            main_after = main_after_metric
            other_after = other_after_metric

        simplified = CoastlineDataset(
            main_gdf=main_after,
            other_gdf=other_after,
            name=name or f"{dataset.name}_simplified_{self._format_tolerance(tolerance_m)}",
        )

        summary = self.build_summary(
            original=dataset,
            simplified=simplified,
            tolerance_m=tolerance_m,
            preserve_topology=preserve_topology,
            metric_crs=str(metric_crs),
        )
        self._log.info(f"Simplification summary: {summary.as_dict()}")

        return simplified

    def build_summary(
        self,
        original: CoastlineDataset,
        simplified: CoastlineDataset,
        tolerance_m: float,
        preserve_topology: bool,
        metric_crs: str,
    ) -> CoastlineSimplificationSummary:
        return CoastlineSimplificationSummary(
            dataset_name=original.name,
            tolerance_m=tolerance_m,
            preserve_topology=preserve_topology,
            source_crs=str(original.crs) if original.crs is not None else None,
            metric_crs=metric_crs,
            main_features_before=len(original.main_gdf),
            other_features_before=len(original.other_gdf),
            main_features_after=len(simplified.main_gdf),
            other_features_after=len(simplified.other_gdf),
            main_vertices_before=self._count_vertices(original.main_gdf),
            other_vertices_before=self._count_vertices(original.other_gdf),
            main_vertices_after=self._count_vertices(simplified.main_gdf),
            other_vertices_after=self._count_vertices(simplified.other_gdf),
        )

    def save(
        self,
        dataset: CoastlineDataset,
        output_dir: str | Path,
        main_filename: str = "simplified_main.geojson",
        other_filename: str = "simplified_other.geojson",
    ) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        main_path = out_dir / main_filename
        other_path = out_dir / other_filename

        dataset.main_gdf.to_file(main_path, driver="GeoJSON")
        dataset.other_gdf.to_file(other_path, driver="GeoJSON")

        self._log.success(
            f"Simplified coastline saved: main={main_path}, other={other_path}"
        )

        return {
            "main": str(main_path),
            "other": str(other_path),
        }

    def simplify_and_save(
        self,
        dataset: CoastlineDataset,
        tolerance_m: float,
        output_dir: str | Path,
        preserve_topology: bool = True,
        name: str | None = None,
        main_filename: str | None = None,
        other_filename: str | None = None,
    ) -> tuple[CoastlineDataset, dict[str, str]]:
        simplified = self.simplify(
            dataset=dataset,
            tolerance_m=tolerance_m,
            preserve_topology=preserve_topology,
            name=name,
        )

        tol_label = self._format_tolerance(tolerance_m)
        saved = self.save(
            dataset=simplified,
            output_dir=output_dir,
            main_filename=main_filename or f"simplified_main_{tol_label}.geojson",
            other_filename=other_filename or f"simplified_other_{tol_label}.geojson",
        )

        return simplified, saved

    def _simplify_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        tolerance_m: float,
        preserve_topology: bool,
    ) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()

        out = gdf.copy()
        out["geometry"] = out.geometry.simplify(
            tolerance=tolerance_m,
            preserve_topology=preserve_topology,
        )
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()

        return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)

    @staticmethod
    def _to_metric(gdf: gpd.GeoDataFrame, metric_crs) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()
        if gdf.crs is None:
            raise ValueError("GeoDataFrame CRS is undefined")
        if str(gdf.crs) == str(metric_crs):
            return gdf.copy()
        return gdf.to_crs(metric_crs)

    @staticmethod
    def _format_tolerance(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)}m"
        return f"{value:.3f}m".replace(".", "_")

    def _count_vertices(self, gdf: gpd.GeoDataFrame) -> int:
        if gdf.empty:
            return 0

        total = 0
        for geom in gdf.geometry:
            total += self._count_geom_vertices(geom)
        return total

    def _count_geom_vertices(self, geom) -> int:
        if geom is None or geom.is_empty:
            return 0

        geom_type = geom.geom_type

        if geom_type == "LineString":
            return len(geom.coords)

        if geom_type == "LinearRing":
            return len(geom.coords)

        if geom_type == "MultiLineString":
            return sum(len(part.coords) for part in geom.geoms if not part.is_empty)

        if geom_type == "Polygon":
            total = len(geom.exterior.coords)
            total += sum(len(ring.coords) for ring in geom.interiors)
            return total

        if geom_type == "MultiPolygon":
            total = 0
            for poly in geom.geoms:
                if poly.is_empty:
                    continue
                total += len(poly.exterior.coords)
                total += sum(len(ring.coords) for ring in poly.interiors)
            return total

        if hasattr(geom, "geoms"):
            return sum(self._count_geom_vertices(part) for part in geom.geoms)

        if hasattr(geom, "coords"):
            return len(geom.coords)

        return 0
