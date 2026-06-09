from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.coastline.domain.models import BoundingBox, CoastlineSummary


class CoastlineDataset:
    """
    Доменная модель набора береговых линий.

    Хранит основную (main) и дополнительную (other) линии,
    вычисляет bbox и собирает сводку. Не содержит логики
    экспорта или извлечения точек.

    Важно:
    - исходная CRS может быть географической (например EPSG:4326);
    - длины для summary всегда считаются в метрической projected CRS;
    - это устраняет warning GeoPandas и даёт физически интерпретируемые значения.
    """

    DEFAULT_CRS = "EPSG:4326"

    def __init__(
        self,
        main_gdf: gpd.GeoDataFrame,
        other_gdf: gpd.GeoDataFrame,
        name: str = "coastline_dataset",
    ) -> None:
        self.name = name
        self._log = logger.bind(cls="CoastlineDataset", name=name)

        self.main_gdf = self._prepare(main_gdf, role="main_coastline")
        self.other_gdf = self._prepare(other_gdf, role="other_coastline")

        self._align_crs()

        self._log.debug(
            f"Initialized: main={len(self.main_gdf)}, other={len(self.other_gdf)}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_geojson(
        cls,
        main_path: str | Path,
        other_path: str | Path,
        name: str = "coastline_dataset",
    ) -> "CoastlineDataset":
        log = logger.bind(cls="CoastlineDataset", name=name)
        log.info(f"Reading main:  {main_path}")
        log.info(f"Reading other: {other_path}")

        return cls(
            main_gdf=gpd.read_file(Path(main_path)),
            other_gdf=gpd.read_file(Path(other_path)),
            name=name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def crs(self):
        return self.main_gdf.crs if self.main_gdf.crs is not None else self.other_gdf.crs

    @property
    def combined_gdf(self) -> gpd.GeoDataFrame:
        combined = pd.concat([self.main_gdf, self.other_gdf], ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=self.crs)

    @property
    def bbox(self) -> BoundingBox:
        b = self.combined_gdf.total_bounds
        return BoundingBox(
            minx=float(b[0]),
            miny=float(b[1]),
            maxx=float(b[2]),
            maxy=float(b[3]),
        )

    @property
    def metric_crs(self):
        """
        CRS для метрических расчётов длины.

        Логика:
        - если текущая CRS уже projected, используем её;
        - если geographic, пытаемся оценить UTM по данным;
        - если не удалось, пробуем main_gdf, other_gdf по отдельности.
        """
        crs = self.crs
        if crs is None:
            raise ValueError("Dataset CRS is undefined")

        if not crs.is_geographic:
            return crs

        combined = self.combined_gdf
        metric = combined.estimate_utm_crs()
        if metric is not None:
            return metric

        if not self.main_gdf.empty:
            metric = self.main_gdf.estimate_utm_crs()
            if metric is not None:
                return metric

        if not self.other_gdf.empty:
            metric = self.other_gdf.estimate_utm_crs()
            if metric is not None:
                return metric

        raise ValueError("Failed to estimate metric CRS for coastline dataset")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> CoastlineSummary:
        metric_crs = self.metric_crs

        ml = self._safe_length_m(self.main_gdf, metric_crs=metric_crs)
        ol = self._safe_length_m(self.other_gdf, metric_crs=metric_crs)

        s = CoastlineSummary(
            name=self.name,
            crs=str(self.crs) if self.crs is not None else None,
            bbox=self.bbox,
            main_feature_count=len(self.main_gdf),
            other_feature_count=len(self.other_gdf),
            total_feature_count=len(self.main_gdf) + len(self.other_gdf),
            main_total_length=ml,
            other_total_length=ol,
            total_length=(ml or 0.0) + (ol or 0.0),
        )

        self._log.info(f"Summary: {s.as_dict()}")
        return s

    def print_summary(self) -> None:
        s = self.summary()

        def _fmt_m(v: float | None) -> str:
            if v is None:
                return "None"
            if v >= 1000:
                return f"{v:.2f} m ({v / 1000:.3f} km)"
            return f"{v:.2f} m"

        print(f"=== {s.name} ===")
        print(f"  CRS           : {s.crs}")
        print(f"  Metric CRS    : {self.metric_crs}")
        print(f"  BBox          : {s.bbox}")
        print(f"  Main features : {s.main_feature_count}")
        print(f"  Other features: {s.other_feature_count}")
        print(f"  Total features: {s.total_feature_count}")
        print(f"  Main length   : {_fmt_m(s.main_total_length)}")
        print(f"  Other length  : {_fmt_m(s.other_total_length)}")
        print(f"  Total length  : {_fmt_m(s.total_length)}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare(self, gdf: gpd.GeoDataFrame, role: str) -> gpd.GeoDataFrame:
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
        mc, oc = self.main_gdf.crs, self.other_gdf.crs

        if mc is None and oc is None:
            self._log.warning(
                f"Both GDFs have no CRS — assuming {self.DEFAULT_CRS}"
            )
            self.main_gdf = self.main_gdf.set_crs(self.DEFAULT_CRS)
            self.other_gdf = self.other_gdf.set_crs(self.DEFAULT_CRS)

        elif mc is None:
            self._log.warning(f"Main CRS missing — assigning {oc}")
            self.main_gdf = self.main_gdf.set_crs(oc)

        elif oc is None:
            self._log.warning(f"Other CRS missing — assigning {mc}")
            self.other_gdf = self.other_gdf.set_crs(mc)

        elif mc != oc:
            self._log.info(f"Reprojecting other {oc} → {mc}")
            self.other_gdf = self.other_gdf.to_crs(mc)

    def _safe_length_m(self, gdf: gpd.GeoDataFrame, metric_crs) -> float | None:
        """
        Безопасный расчёт общей длины в МЕТРАХ.

        Если входной gdf в geographic CRS, он временно перепроецируется
        в projected metric CRS перед вычислением длины.
        """
        if gdf.empty:
            return 0.0

        try:
            if gdf.crs is None:
                raise ValueError("GeoDataFrame has no CRS")

            if gdf.crs.is_geographic:
                gdf_metric = gdf.to_crs(metric_crs)
            else:
                gdf_metric = gdf

            return float(gdf_metric.geometry.length.sum())

        except Exception:
            self._log.exception("Failed to calculate geometry lengths in meters")
            return None

    # ------------------------------------------------------------------
    # Export (delegated to strategy)
    # ------------------------------------------------------------------

    def export(
        self,
        strategy: "CoastlineExportStrategy",
        output_path: "str | Path",
    ) -> "Path":
        from src.coastline.exporters.CoastlineExportStrategy import (
            CoastlineExportStrategy,
        )

        self._log.info(f"Export via {strategy.__class__.__name__} → {output_path}")
        return strategy.export(self, output_path)
