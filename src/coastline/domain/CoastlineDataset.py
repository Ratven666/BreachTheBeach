from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import GeometryCollection, LineString, MultiLineString, Point, shortest_line, line_merge
from shapely.geometry import box
from shapely.ops import snap, unary_union

from src.coastline.domain.models import BoundingBox, CoastlineSummary


@dataclass(frozen=True)
class _MergeContext:
    inner: "CoastlineDataset"
    outer: "CoastlineDataset"
    work_crs: object
    inner_bbox_polygon: object


@dataclass(frozen=True)
class _OpenEndpoint:
    point: Point
    source_idx: int
    side: str


class CoastlineDataset:
    """
    Доменная модель набора береговых линий.

    Поддерживает:
    - основной слой main_gdf;
    - опциональный слой other_gdf, который может быть пустым.

    Важно:
    - если other_gdf не передан, создаётся пустой корректный GeoDataFrame;
    - main и other обрабатываются независимо;
    - merge выполняется в projected CRS;
    - экспорт поддерживает как раздельные слои, так и объединённый слой.
    """

    DEFAULT_CRS = "EPSG:4326"

    def __init__(
        self,
        main_gdf: gpd.GeoDataFrame,
        other_gdf: gpd.GeoDataFrame | None = None,
        name: str = "coastline_dataset",
    ) -> None:
        self.name = name
        self._log = logger.bind(cls="CoastlineDataset", name=name)

        self.main_gdf = self._prepare(main_gdf, role="main_coastline")
        self.other_gdf = self._prepare_optional_other(other_gdf)

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
        other_path: str | Path | None = None,
        name: str = "coastline_dataset",
    ) -> "CoastlineDataset":
        log = logger.bind(cls="CoastlineDataset", name=name)
        log.info(f"Reading main:  {main_path}")

        main_gdf = gpd.read_file(Path(main_path))

        other_gdf = None
        if other_path is not None:
            log.info(f"Reading other: {other_path}")
            other_gdf = gpd.read_file(Path(other_path))
        else:
            log.info("Reading other: skipped (None)")

        return cls(
            main_gdf=main_gdf,
            other_gdf=other_gdf,
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
        frames = [self.main_gdf]
        if self.other_gdf is not None and not self.other_gdf.empty:
            frames.append(self.other_gdf)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["geometry"])
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=self.crs)

    @property
    def bbox(self) -> BoundingBox:
        if self.combined_gdf.empty:
            raise ValueError("Cannot compute bbox for empty coastline dataset")

        b = self.combined_gdf.total_bounds
        return BoundingBox(
            minx=float(b[0]),
            miny=float(b[1]),
            maxx=float(b[2]),
            maxy=float(b[3]),
        )

    @property
    def bbox_polygon(self):
        if self.combined_gdf.empty:
            raise ValueError("Cannot compute bbox polygon for empty coastline dataset")

        b = self.combined_gdf.total_bounds
        return box(float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def metric_crs(self):
        crs = self.crs
        if crs is None:
            raise ValueError("Dataset CRS is undefined")

        if not crs.is_geographic:
            return crs

        combined = self.combined_gdf
        if not combined.empty:
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
    # Merge
    # ------------------------------------------------------------------

    def merge_with_replacement(
        self,
        other: "CoastlineDataset",
        *,
        prefer_inner: str = "smaller_bbox",
        close_gaps: bool = True,
        snap_tolerance: float | None = None,
        max_gap_distance: float | None = None,
        name: str | None = None,
    ) -> "CoastlineDataset":
        ctx = self._build_merge_context(other, prefer_inner=prefer_inner)

        self._log.info(
            f"Merging datasets: inner={ctx.inner.name}, outer={ctx.outer.name}, crs={ctx.work_crs}"
        )

        merged_main = self._merge_single_role(
            outer_gdf=ctx.outer.main_gdf,
            inner_gdf=ctx.inner.main_gdf,
            role="main_coastline",
            bbox_polygon=ctx.inner_bbox_polygon,
            close_gaps=close_gaps,
            snap_tolerance=snap_tolerance,
            max_gap_distance=max_gap_distance,
        )

        merged_other = self._merge_single_role(
            outer_gdf=ctx.outer.other_gdf,
            inner_gdf=ctx.inner.other_gdf,
            role="other_coastline",
            bbox_polygon=ctx.inner_bbox_polygon,
            close_gaps=close_gaps,
            snap_tolerance=snap_tolerance,
            max_gap_distance=max_gap_distance,
        )

        return CoastlineDataset(
            main_gdf=merged_main,
            other_gdf=merged_other,
            name=name or f"{self.name}_merged_{other.name}",
        )

    def _merge_single_role(
        self,
        *,
        outer_gdf: gpd.GeoDataFrame,
        inner_gdf: gpd.GeoDataFrame,
        role: str,
        bbox_polygon,
        close_gaps: bool,
        snap_tolerance: float | None,
        max_gap_distance: float | None,
    ) -> gpd.GeoDataFrame:
        if outer_gdf.empty and inner_gdf.empty:
            return self._empty_like(role=role, crs=self.crs)

        base_crs = outer_gdf.crs if outer_gdf.crs is not None else inner_gdf.crs

        outer_kept = self._erase_bbox_area(outer_gdf, bbox_polygon)
        inner_insert = self._clip_to_polygon(inner_gdf, bbox_polygon)

        merged = gpd.GeoDataFrame(
            pd.concat([outer_kept, inner_insert], ignore_index=True),
            geometry="geometry",
            crs=base_crs,
        )
        merged = self._clean_geometries(merged)

        if close_gaps and not merged.empty:
            merged = self._close_gaps_by_nearest_endpoints(
                merged,
                bbox_polygon,
                role=role,
                snap_tolerance=snap_tolerance,
                max_gap_distance=max_gap_distance,
            )

        merged = self._normalize_linework(merged, role=role)
        return merged

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
        else:
            gdf["coastline_role"] = role

        return gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)

    def _prepare_optional_other(self, other_gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
        if other_gdf is None:
            return self._empty_like(role="other_coastline", crs=self.main_gdf.crs)

        return self._prepare(other_gdf, role="other_coastline")

    def _align_crs(self) -> None:
        mc, oc = self.main_gdf.crs, self.other_gdf.crs

        if mc is None and oc is None:
            self._log.warning(f"Both GDFs have no CRS — assuming {self.DEFAULT_CRS}")
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

    def _build_merge_context(
        self,
        other: "CoastlineDataset",
        *,
        prefer_inner: str = "smaller_bbox",
    ) -> _MergeContext:
        left = self._as_working_dataset()
        right = other._reprojected_copy(left.crs)

        left_bbox = left.bbox_polygon
        right_bbox = right.bbox_polygon

        left_area = float(left_bbox.area)
        right_area = float(right_bbox.area)

        if prefer_inner == "smaller_bbox":
            inner, outer = (left, right) if left_area <= right_area else (right, left)
        else:
            raise ValueError(f"Unsupported prefer_inner: {prefer_inner}")

        inner_bbox_polygon = inner.bbox_polygon

        if not outer.bbox_polygon.contains(inner_bbox_polygon) and not outer.bbox_polygon.equals(inner_bbox_polygon):
            raise ValueError("Inner dataset bbox is not fully contained in outer dataset bbox")

        return _MergeContext(
            inner=inner,
            outer=outer,
            work_crs=left.crs,
            inner_bbox_polygon=inner_bbox_polygon,
        )

    def _as_working_dataset(self) -> "CoastlineDataset":
        crs = self.metric_crs if self.crs is not None and self.crs.is_geographic else self.crs
        return self._reprojected_copy(crs)

    def _reprojected_copy(self, crs) -> "CoastlineDataset":
        if crs is None:
            raise ValueError("Target CRS is undefined")

        main = self.main_gdf.copy()
        other = self.other_gdf.copy()

        if main.crs != crs:
            main = main.to_crs(crs)

        if other.crs != crs:
            other = other.to_crs(crs)

        return CoastlineDataset(main_gdf=main, other_gdf=other, name=self.name)

    def _erase_bbox_area(
        self,
        gdf: gpd.GeoDataFrame,
        mask_polygon,
    ) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()

        out = gdf.copy()
        out["geometry"] = out.geometry.apply(lambda geom: geom.difference(mask_polygon))
        out = self._clean_geometries(out)
        return out

    def _clip_to_polygon(
        self,
        gdf: gpd.GeoDataFrame,
        polygon,
    ) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()

        mask = gpd.GeoDataFrame({"geometry": [polygon]}, crs=gdf.crs)
        clipped = gpd.clip(gdf, mask, keep_geom_type=False)
        clipped = self._clean_geometries(clipped)
        return clipped

    def _close_gaps_by_nearest_endpoints(
        self,
        gdf: gpd.GeoDataFrame,
        bbox_polygon,
        *,
        role: str,
        snap_tolerance: float | None,
        max_gap_distance: float | None,
    ) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()

        tol = snap_tolerance if snap_tolerance is not None else self._default_snap_tolerance(gdf.crs)
        max_gap = max_gap_distance if max_gap_distance is not None else max(tol * 20.0, 10.0)

        endpoints = self._collect_open_endpoints_near_bbox(
            gdf,
            bbox_polygon,
            tol=tol,
        )

        if len(endpoints) < 2:
            self._log.debug(f"{role}: not enough open endpoints for closing")
            return gdf

        bridges = self._build_nearest_endpoint_bridges(
            endpoints,
            bbox_polygon=bbox_polygon,
            max_gap_distance=max_gap,
            tol=tol,
        )

        if not bridges:
            self._log.debug(f"{role}: no valid bridges created")
            return gdf

        self._log.info(f"{role}: created {len(bridges)} nearest bridge segment(s)")

        bridge_gdf = gpd.GeoDataFrame(
            {
                "coastline_role": [role] * len(bridges),
                "geometry": bridges,
            },
            geometry="geometry",
            crs=gdf.crs,
        )

        out = gpd.GeoDataFrame(
            pd.concat([gdf, bridge_gdf], ignore_index=True),
            geometry="geometry",
            crs=gdf.crs,
        )
        return self._clean_geometries(out)

    def _collect_open_endpoints_near_bbox(
        self,
        gdf: gpd.GeoDataFrame,
        bbox_polygon,
        *,
        tol: float,
    ) -> list[_OpenEndpoint]:
        boundary = bbox_polygon.boundary
        raw: list[_OpenEndpoint] = []

        for idx, geom in enumerate(gdf.geometry):
            for line in self._iter_lines(geom):
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                p0 = Point(coords[0])
                p1 = Point(coords[-1])

                if p0.distance(boundary) <= tol:
                    raw.append(
                        _OpenEndpoint(
                            point=self._snap_point_to_bbox_boundary(p0, bbox_polygon, tol),
                            source_idx=idx,
                            side=self._classify_bbox_side(p0, bbox_polygon, tol),
                        )
                    )

                if p1.distance(boundary) <= tol:
                    raw.append(
                        _OpenEndpoint(
                            point=self._snap_point_to_bbox_boundary(p1, bbox_polygon, tol),
                            source_idx=idx,
                            side=self._classify_bbox_side(p1, bbox_polygon, tol),
                        )
                    )

        raw = self._deduplicate_open_endpoints(raw, tol=tol)

        open_points: list[_OpenEndpoint] = []
        for i, ep in enumerate(raw):
            has_neighbor = False
            for j, other in enumerate(raw):
                if i == j:
                    continue
                if ep.source_idx == other.source_idx:
                    continue
                if ep.point.distance(other.point) <= tol:
                    has_neighbor = True
                    break
            if not has_neighbor:
                open_points.append(ep)

        return open_points

    def _build_nearest_endpoint_bridges(
        self,
        endpoints: list[_OpenEndpoint],
        *,
        bbox_polygon,
        max_gap_distance: float,
        tol: float,
    ) -> list[LineString]:
        bridges: list[LineString] = []
        used: set[int] = set()

        for i, ep in enumerate(endpoints):
            if i in used:
                continue

            best_j = None
            best_line = None
            best_dist = None

            for j, other in enumerate(endpoints):
                if j <= i or j in used:
                    continue
                if ep.source_idx == other.source_idx:
                    continue

                candidate = shortest_line(ep.point, other.point)
                dist = float(candidate.length)

                if dist <= tol:
                    continue
                if dist > max_gap_distance:
                    continue
                if not self._bridge_is_valid(candidate, bbox_polygon, tol=tol):
                    continue

                if best_dist is None or dist < best_dist:
                    best_j = j
                    best_line = candidate
                    best_dist = dist

            if best_j is not None and best_line is not None:
                bridges.append(best_line)
                used.add(i)
                used.add(best_j)

        return bridges

    def _bridge_is_valid(
        self,
        bridge: LineString,
        bbox_polygon,
        *,
        tol: float,
    ) -> bool:
        if bridge.is_empty or len(bridge.coords) < 2:
            return False

        if bridge.length <= tol:
            return False

        inner = bbox_polygon.buffer(-tol * 0.25) if tol > 0 else bbox_polygon
        if not inner.is_empty and bridge.crosses(inner):
            return False
        if not inner.is_empty and bridge.within(inner):
            return False

        return True

    def _snap_point_to_bbox_boundary(
        self,
        point: Point,
        bbox_polygon,
        tol: float,
    ) -> Point:
        snapped = snap(point, bbox_polygon.boundary, tol)
        return snapped if isinstance(snapped, Point) else point

    def _classify_bbox_side(
        self,
        point: Point,
        bbox_polygon,
        tol: float,
    ) -> str:
        minx, miny, maxx, maxy = bbox_polygon.bounds
        x, y = point.x, point.y

        if abs(x - minx) <= tol:
            return "left"
        if abs(x - maxx) <= tol:
            return "right"
        if abs(y - miny) <= tol:
            return "bottom"
        if abs(y - maxy) <= tol:
            return "top"
        return "unknown"

    def _normalize_linework(
        self,
        gdf: gpd.GeoDataFrame,
        *,
        role: str,
    ) -> gpd.GeoDataFrame:
        if gdf.empty:
            return self._empty_like(role=role, crs=gdf.crs)

        merged_geom = unary_union(gdf.geometry.values)
        merged_geom = line_merge(merged_geom)

        lines = list(self._iter_lines(merged_geom))
        if not lines:
            return self._empty_like(role=role, crs=gdf.crs)

        out = gpd.GeoDataFrame(
            {
                "coastline_role": [role] * len(lines),
                "geometry": lines,
            },
            geometry="geometry",
            crs=gdf.crs,
        )
        return self._clean_geometries(out)

    def _empty_like(self, *, role: str, crs) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"coastline_role": pd.Series(dtype="object"), "geometry": pd.Series(dtype="object")},
            geometry="geometry",
            crs=crs,
        )

    def _clean_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()

        exploded_rows: list[dict] = []

        for _, row in gdf.iterrows():
            base = row.drop(labels=["geometry"]).to_dict()
            for line in self._iter_lines(row.geometry):
                if line.is_empty:
                    continue
                if len(line.coords) < 2:
                    continue

                item = dict(base)
                item["geometry"] = line
                exploded_rows.append(item)

        if not exploded_rows:
            cols = [c for c in gdf.columns if c != "geometry"]
            empty = {c: pd.Series(dtype="object") for c in cols}
            empty["geometry"] = pd.Series(dtype="object")
            return gpd.GeoDataFrame(empty, geometry="geometry", crs=gdf.crs)

        return gpd.GeoDataFrame(exploded_rows, geometry="geometry", crs=gdf.crs)

    def _iter_lines(self, geom) -> Iterable[LineString]:
        if geom is None or geom.is_empty:
            return

        if isinstance(geom, LineString):
            yield geom
            return

        if isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if not part.is_empty:
                    yield part
            return

        if isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                yield from self._iter_lines(part)
            return

        if hasattr(geom, "boundary") and geom.geom_type in {"Polygon", "MultiPolygon"}:
            return

    def _deduplicate_open_endpoints(
        self,
        endpoints: list[_OpenEndpoint],
        *,
        tol: float,
    ) -> list[_OpenEndpoint]:
        if not endpoints:
            return []

        unique: list[_OpenEndpoint] = []

        for ep in endpoints:
            exists = False
            for u in unique:
                if ep.source_idx == u.source_idx and ep.point.distance(u.point) <= tol:
                    exists = True
                    break
            if not exists:
                unique.append(ep)

        return unique

    def _default_snap_tolerance(self, crs) -> float:
        try:
            if crs is not None and not crs.is_geographic:
                return 1.0
        except Exception:
            pass
        return 1e-6

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_combined_geojson(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.combined_gdf.to_file(output_path, driver="GeoJSON")
        self._log.info(f"Exported combined dataset to {output_path}")
        return output_path

    def export_split_geojson(
        self,
        main_output_path: str | Path,
        other_output_path: str | Path | None = None,
    ) -> tuple[Path, Path | None]:
        main_output_path = Path(main_output_path)
        main_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.main_gdf.to_file(main_output_path, driver="GeoJSON")
        self._log.info(f"Exported main dataset to {main_output_path}")

        other_path: Path | None = None
        if other_output_path is not None:
            other_path = Path(other_output_path)
            other_path.parent.mkdir(parents=True, exist_ok=True)
            self.other_gdf.to_file(other_path, driver="GeoJSON")
            self._log.info(f"Exported other dataset to {other_path}")

        return main_output_path, other_path

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


if __name__ == "__main__":
    from loguru import logger

    OUTER_MAIN_PATH = r"data/nvrsk_black_sea_main_coastline_fast.geojson"
    OUTER_OTHER_PATH = r"data/nvrsk_black_sea_other_lines_fast.geojson"

    INNER_MAIN_PATH = r"data/main_coastline.geojson"
    INNER_OTHER_PATH = r"data/other_lines.geojson"

    OUTPUT_MAIN_PATH = r"output/merged_main.geojson"
    OUTPUT_OTHER_PATH = r"output/merged_other.geojson"
    OUTPUT_COMBINED_PATH = r"output/merged_dataset.geojson"

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True)

    ds_outer = CoastlineDataset.from_geojson(
        main_path=OUTER_MAIN_PATH,
        other_path=OUTER_OTHER_PATH,
        name="outer_dataset",
    )

    ds_inner = CoastlineDataset.from_geojson(
        main_path=INNER_MAIN_PATH,
        other_path=INNER_OTHER_PATH,
        name="inner_dataset",
    )

    merged = ds_outer.merge_with_replacement(
        ds_inner,
        close_gaps=True,
        snap_tolerance=2.0,
        max_gap_distance=100.0,
        name="merged_dataset",
    )

    merged.print_summary()

    merged.export_split_geojson(
        main_output_path=OUTPUT_MAIN_PATH,
        other_output_path=OUTPUT_OTHER_PATH,
    )

    merged.export_combined_geojson(OUTPUT_COMBINED_PATH)

    logger.success("Export completed")
