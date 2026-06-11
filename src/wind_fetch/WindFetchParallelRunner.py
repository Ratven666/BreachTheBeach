from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Iterable, Sequence

import geopandas as gpd
from loguru import logger
from shapely import from_wkb
from shapely.geometry import Point
from shapely.wkb import dumps as to_wkb

from .CoastlineSpatialIndex import CoastlineSpatialIndex
from .WindFetchConfig import WindFetchConfig
from .geometry_utils import (
    build_geodesic_linestring,
    extract_points_from_intersection,
    geodesic_distance_m,
    geodesic_forward_point,
)
from .models import MultiDirectionFetchResult, WindFetchPaths


_WORKER_COASTLINE_GEOMS = None
_WORKER_INDEX = None
_WORKER_CONFIG = None


def _worker_init(
    coastline_wkb: list[bytes],
    config_dict: dict,
) -> None:
    global _WORKER_COASTLINE_GEOMS, _WORKER_INDEX, _WORKER_CONFIG

    geoms = [from_wkb(wkb) for wkb in coastline_wkb]
    coastline_gdf = gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs="EPSG:4326")

    _WORKER_COASTLINE_GEOMS = geoms
    _WORKER_INDEX = CoastlineSpatialIndex(coastline_gdf)
    _WORKER_CONFIG = WindFetchConfig(**config_dict)

    logger.debug(
        f"Worker initialized: coastline_features={len(_WORKER_COASTLINE_GEOMS)}"
    )


def _normalize_azimuths(
    azimuths: Iterable[float] | None,
) -> list[float]:
    if azimuths is None:
        return [float(v) for v in range(360)]

    normalized = sorted({float(v) % 360.0 for v in azimuths})
    if not normalized:
        raise ValueError("Azimuth list is empty")

    return normalized


def _find_first_intersection(
    ray,
    start_lon: float,
    start_lat: float,
):
    global _WORKER_INDEX

    candidates = _WORKER_INDEX.query(ray)
    if not candidates:
        return None

    nearest_point = None
    nearest_dist = None

    for coastline in candidates:
        if coastline is None or coastline.is_empty:
            continue

        if not coastline.intersects(ray):
            continue

        inter = ray.intersection(coastline)
        if inter.is_empty:
            continue

        points = extract_points_from_intersection(inter)
        if not points:
            continue

        for pt in points:
            dist = geodesic_distance_m(
                start_lon,
                start_lat,
                float(pt.x),
                float(pt.y),
            )

            if dist <= 1e-9:
                continue

            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_point = pt

    return nearest_point


def _process_chunk(
    chunk_points: list[tuple[int, float, float]],
    azimuths: list[float],
    offset_m: float,
) -> list[dict]:
    global _WORKER_CONFIG

    results: list[dict] = []
    append_result = results.append

    for point_id, source_lon, source_lat in chunk_points:
        for direction_id, azimuth_deg in enumerate(azimuths, start=1):
            start_lon, start_lat = geodesic_forward_point(
                source_lon,
                source_lat,
                azimuth_deg,
                offset_m,
            )

            ray = build_geodesic_linestring(
                lon=start_lon,
                lat=start_lat,
                azimuth_deg=azimuth_deg,
                total_length_m=_WORKER_CONFIG.default_fetch_m,
                step_m=_WORKER_CONFIG.geodesic_step_m,
                max_segments=_WORKER_CONFIG.max_segments_per_ray,
            )

            hit_point = _find_first_intersection(
                ray=ray,
                start_lon=start_lon,
                start_lat=start_lat,
            )

            if hit_point is None:
                append_result(
                    {
                        "point_id": point_id,
                        "direction_id": direction_id,
                        "azimuth_deg": azimuth_deg,
                        "source_point_lon": source_lon,
                        "source_point_lat": source_lat,
                        "start_point_lon": start_lon,
                        "start_point_lat": start_lat,
                        "fetch_length_m": _WORKER_CONFIG.default_fetch_m,
                        "hit_found": False,
                        "hit_lon": None,
                        "hit_lat": None,
                        "used_default_value": True,
                    }
                )
                continue

            hit_lon = float(hit_point.x)
            hit_lat = float(hit_point.y)
            fetch_length_m = geodesic_distance_m(
                start_lon,
                start_lat,
                hit_lon,
                hit_lat,
            )

            append_result(
                {
                    "point_id": point_id,
                    "direction_id": direction_id,
                    "azimuth_deg": azimuth_deg,
                    "source_point_lon": source_lon,
                    "source_point_lat": source_lat,
                    "start_point_lon": start_lon,
                    "start_point_lat": start_lat,
                    "fetch_length_m": fetch_length_m,
                    "hit_found": True,
                    "hit_lon": hit_lon,
                    "hit_lat": hit_lat,
                    "used_default_value": False,
                }
            )

    return results


class WindFetchParallelRunner:
    """
    Параллельный раннер для MultiDirectionFetchCalculator.

    Делит набор точек на чанки и считает каждый chunk в отдельном процессе.
    """

    def __init__(
        self,
        paths: WindFetchPaths,
        config: WindFetchConfig | None = None,
    ) -> None:
        self.paths = paths
        self.config = config or WindFetchConfig()

        self.points_gdf = gpd.read_file(paths.points_with_normals_path)
        if self.points_gdf.crs is None:
            self.points_gdf = self.points_gdf.set_crs("EPSG:4326")
        elif str(self.points_gdf.crs) != "EPSG:4326":
            self.points_gdf = self.points_gdf.to_crs("EPSG:4326")

        coastline_main = gpd.read_file(paths.main_coastline_path)
        coastline_frames = [coastline_main]

        if paths.other_coastline_path is not None:
            coastline_other = gpd.read_file(paths.other_coastline_path)
            coastline_frames.append(coastline_other)

        coastline_gdf = gpd.GeoDataFrame(
            gpd.pd.concat(coastline_frames, ignore_index=True),
            geometry="geometry",
            crs=coastline_frames[0].crs,
        )

        if coastline_gdf.crs is None:
            coastline_gdf = coastline_gdf.set_crs("EPSG:4326")
        elif str(coastline_gdf.crs) != "EPSG:4326":
            coastline_gdf = coastline_gdf.to_crs("EPSG:4326")

        coastline_gdf = coastline_gdf[
            coastline_gdf.geometry.notna() & ~coastline_gdf.geometry.is_empty
        ].copy()

        self.coastline_wkb = [to_wkb(geom) for geom in coastline_gdf.geometry.values]

        logger.info(
            f"WindFetchParallelRunner initialized: "
            f"points={len(self.points_gdf)}, coastline_features={len(self.coastline_wkb)}"
        )

    def calculate_multi_direction(
        self,
        azimuths: Iterable[float] | None = None,
        offset_m: float | None = None,
        max_workers: int | None = None,
        chunk_size: int = 250,
    ) -> list[MultiDirectionFetchResult]:
        azimuth_list = _normalize_azimuths(azimuths)
        offset = self.config.default_offset_m if offset_m is None else float(offset_m)

        points = self._extract_points()
        point_chunks = self._chunk_points(points, chunk_size=chunk_size)

        workers = max_workers or os.cpu_count() or 1

        logger.info(
            f"Starting parallel multi-direction fetch: "
            f"points={len(points)}, directions={len(azimuth_list)}, "
            f"workers={workers}, chunks={len(point_chunks)}, chunk_size={chunk_size}"
        )

        config_dict = asdict(self.config)
        raw_results: list[dict] = []

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(self.coastline_wkb, config_dict),
        ) as executor:
            for chunk_result in executor.map(
                _process_chunk,
                point_chunks,
                [azimuth_list] * len(point_chunks),
                [offset] * len(point_chunks),
                chunksize=1,
            ):
                raw_results.extend(chunk_result)

        results = [MultiDirectionFetchResult(**item) for item in raw_results]

        logger.success(
            f"Parallel multi-direction fetch completed: results={len(results)}"
        )
        return results

    def _extract_points(self) -> list[tuple[int, float, float]]:
        points: list[tuple[int, float, float]] = []

        for idx, geom in enumerate(self.points_gdf.geometry.values, start=1):
            if geom is None or geom.is_empty:
                continue

            point = geom
            if not isinstance(point, Point):
                continue

            points.append((idx, float(point.x), float(point.y)))

        return points

    @staticmethod
    def _chunk_points(
        points: Sequence[tuple[int, float, float]],
        chunk_size: int,
    ) -> list[list[tuple[int, float, float]]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        return [
            list(points[i : i + chunk_size])
            for i in range(0, len(points), chunk_size)
        ]
