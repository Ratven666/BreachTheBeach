from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
)
from shapely.ops import substring


@dataclass(frozen=True, slots=True)
class MoranLineExportStats:
    feature_count: int
    matched_segment_count: int
    missing_segment_count: int
    degenerate_count: int
    output_path: Path
    coastline_crs: str
    metric_crs: str


def _iter_line_parts(geometry) -> Iterator[LineString]:
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, LineString):
        if geometry.length > 0.0:
            yield geometry
        return

    if isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            if isinstance(part, LineString) and part.length > 0.0:
                yield part
        return

    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _iter_line_parts(item)
        return


def _load_coastline_segments(
    coastline_geojson: Path,
    metric_crs: str | None,
) -> tuple[gpd.GeoDataFrame, CRS, str]:
    coastline = gpd.read_file(coastline_geojson)

    if coastline.empty:
        raise ValueError(f"Coastline file is empty: {coastline_geojson}")

    if coastline.crs is None:
        coastline = coastline.set_crs("EPSG:4326")

    coastline_crs = CRS.from_user_input(coastline.crs).to_string()

    records: list[dict] = []
    segment_rank = 0

    for feature_index, geometry in enumerate(coastline.geometry):
        for part_index, part in enumerate(_iter_line_parts(geometry)):
            records.append(
                {
                    "segment_id": (
                        f"feature_{feature_index:06d}"
                        f"_part_{part_index:04d}"
                    ),
                    "segment_rank": segment_rank,
                    "geometry": part,
                }
            )
            segment_rank += 1

    if not records:
        raise ValueError(
            "Coastline does not contain non-empty line geometries"
        )

    segments = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=coastline.crs,
    )

    if metric_crs is None:
        estimated = segments.to_crs("EPSG:4326").estimate_utm_crs()
        if estimated is None:
            raise ValueError(
                "Unable to estimate projected CRS from coastline. "
                "Pass metric_crs explicitly."
            )
        target_crs = CRS.from_user_input(estimated)
    else:
        target_crs = CRS.from_user_input(metric_crs)

    if target_crs.is_geographic:
        raise ValueError(
            f"metric_crs must be projected, got {target_crs}"
        )

    return segments.to_crs(target_crs), target_crs, coastline_crs


def _normalize_point_export(
    points_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    required = {
        "point_id",
        "segment_id",
        "order_in_segment",
        "chainage_m",
        "segment_length_m",
        "is_closed",
    }

    missing = sorted(required - set(points_gdf.columns))
    if missing:
        raise ValueError(
            "Point export is missing required fields: "
            f"{missing}"
        )

    if points_gdf.empty:
        raise ValueError("Point export is empty")

    gdf = points_gdf.copy()

    gdf["point_id"] = gdf["point_id"].astype(int)
    gdf["segment_id"] = gdf["segment_id"].astype(str)
    gdf["order_in_segment"] = gdf["order_in_segment"].astype(int)
    gdf["chainage_m"] = gdf["chainage_m"].astype(float)
    gdf["segment_length_m"] = gdf["segment_length_m"].astype(float)
    gdf["is_closed"] = gdf["is_closed"].astype(int)

    if "snap_ok" in gdf.columns:
        gdf = gdf[gdf["snap_ok"].astype(bool)].copy()

    gdf = gdf.sort_values(
        ["segment_id", "order_in_segment", "chainage_m", "point_id"]
    ).reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No valid points remain after filtering")

    return gdf


def _segment_bounds_open(
    chainages: np.ndarray,
    length: float,
) -> list[tuple[float, float]]:
    n = chainages.size
    bounds: list[tuple[float, float]] = []

    for i in range(n):
        left = 0.0 if i == 0 else 0.5 * (chainages[i - 1] + chainages[i])
        right = float(length) if i == n - 1 else 0.5 * (chainages[i] + chainages[i + 1])

        left = max(0.0, min(float(length), float(left)))
        right = max(0.0, min(float(length), float(right)))

        if right < left:
            left, right = right, left

        bounds.append((left, right))

    return bounds


def _segment_bounds_closed(
    chainages: np.ndarray,
    length: float,
) -> list[tuple[float, float]]:
    n = chainages.size

    if n == 1:
        return [(0.0, float(length))]

    extended = np.concatenate([chainages, [chainages[0] + float(length)]])
    mids = 0.5 * (extended[:-1] + extended[1:])

    bounds: list[tuple[float, float]] = []

    for i in range(n):
        left = mids[i - 1] if i > 0 else mids[-1] - float(length)
        right = mids[i]

        left = float(left % float(length))
        right = float(right % float(length))

        bounds.append((left, right))

    return bounds


def _as_linear_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 1.0e-12 else []

    if isinstance(geometry, MultiLineString):
        return [
            part
            for part in geometry.geoms
            if isinstance(part, LineString) and part.length > 1.0e-12
        ]

    return []


def _cut_segment(
    line: LineString,
    start_m: float,
    end_m: float,
    is_closed: bool,
) -> LineString | MultiLineString | None:
    length = float(line.length)

    if length <= 1.0e-12:
        return None

    start_m = float(max(0.0, min(length, start_m)))
    end_m = float(max(0.0, min(length, end_m)))

    if not is_closed:
        if end_m - start_m <= 1.0e-12:
            return None

        geom = substring(line, start_m, end_m)
        parts = _as_linear_parts(geom)

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return MultiLineString(parts)

    if abs(start_m - end_m) <= 1.0e-12:
        return line

    if end_m > start_m:
        geom = substring(line, start_m, end_m)
        parts = _as_linear_parts(geom)

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return MultiLineString(parts)

    part1 = substring(line, start_m, length)
    part2 = substring(line, 0.0, end_m)

    parts: list[LineString] = []
    parts.extend(_as_linear_parts(part1))
    parts.extend(_as_linear_parts(part2))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiLineString(parts)


def _drop_geometry(row) -> dict:
    data = row.to_dict()
    data.pop("geometry", None)
    return data


def build_moran_line_segments(
    point_export_geojson: str | Path,
    coastline_geojson: str | Path,
    output_geojson: str | Path,
    *,
    metric_crs: str | None = None,
    strict_segment_match: bool = True,
) -> MoranLineExportStats:
    point_export_geojson = Path(point_export_geojson).expanduser().resolve()
    coastline_geojson = Path(coastline_geojson).expanduser().resolve()
    output_geojson = Path(output_geojson).expanduser().resolve()

    if not point_export_geojson.is_file():
        raise FileNotFoundError(
            f"Point export not found: {point_export_geojson}"
        )

    if not coastline_geojson.is_file():
        raise FileNotFoundError(
            f"Coastline not found: {coastline_geojson}"
        )

    points = gpd.read_file(point_export_geojson)
    points = _normalize_point_export(points)

    segments_gdf, target_crs, coastline_crs = _load_coastline_segments(
        coastline_geojson,
        metric_crs,
    )

    segment_lookup = {
        str(row.segment_id): row.geometry
        for row in segments_gdf.itertuples()
    }

    point_segment_ids = set(points["segment_id"].astype(str).unique())
    coast_segment_ids = set(segment_lookup.keys())

    missing_segment_ids = sorted(point_segment_ids - coast_segment_ids)

    if missing_segment_ids and strict_segment_match:
        raise ValueError(
            "Some segment_id values from point export are missing in coastline: "
            f"{missing_segment_ids[:20]}"
        )

    records: list[dict] = []
    matched_segment_count = 0
    degenerate_count = 0

    for segment_id, group in points.groupby("segment_id", sort=True):
        line = segment_lookup.get(str(segment_id))
        if line is None:
            continue

        matched_segment_count += 1

        group = group.sort_values(
            ["order_in_segment", "chainage_m", "point_id"]
        ).reset_index(drop=True)

        if group.empty:
            continue

        is_closed_points = bool(int(group.iloc[0]["is_closed"]))
        line_length_geom = float(line.length)
        line_length_points = float(group.iloc[0]["segment_length_m"])

        if abs(line_length_geom - line_length_points) > 1.0:
            # Это сильный признак того, что структура линии не совпадает с той,
            # что использовалась на этапе 5.6.
            if strict_segment_match:
                raise ValueError(
                    "Segment length mismatch for "
                    f"{segment_id}: points={line_length_points}, "
                    f"coastline={line_length_geom}"
                )

        chainages = group["chainage_m"].to_numpy(dtype=float)
        chainages = np.clip(chainages, 0.0, line_length_geom)

        if is_closed_points:
            bounds = _segment_bounds_closed(chainages, line_length_geom)
        else:
            bounds = _segment_bounds_open(chainages, line_length_geom)

        for row_index, (_, row) in enumerate(group.iterrows()):
            start_m, end_m = bounds[row_index]

            geom = _cut_segment(
                line=line,
                start_m=float(start_m),
                end_m=float(end_m),
                is_closed=is_closed_points,
            )

            if geom is None or geom.is_empty or float(geom.length) <= 1.0e-9:
                degenerate_count += 1
                continue

            properties = _drop_geometry(row)
            properties.update(
                {
                    "line_segment_id": f"{segment_id}_{int(row['point_id'])}",
                    "source_segment_id": str(segment_id),
                    "segment_start_m": float(start_m),
                    "segment_end_m": float(end_m),
                    "segment_geom_length_m": float(geom.length),
                    "coastline_part_length_m": float(line_length_geom),
                    "segment_length_match_error_m": float(
                        line_length_geom - line_length_points
                    ),
                }
            )

            records.append(
                {
                    **properties,
                    "geometry": geom,
                }
            )

    if not records:
        raise ValueError(
            "No line segments were generated from point export"
        )

    result = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=target_crs,
    ).to_crs("EPSG:4326")

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_geojson, driver="GeoJSON")

    return MoranLineExportStats(
        feature_count=len(result),
        matched_segment_count=matched_segment_count,
        missing_segment_count=len(missing_segment_ids),
        degenerate_count=degenerate_count,
        output_path=output_geojson,
        coastline_crs=coastline_crs,
        metric_crs=target_crs.to_string(),
    )
