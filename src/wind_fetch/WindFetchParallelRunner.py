# src/wind_fetch/WindFetchParallelRunner.py
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyproj import Geod
from shapely.geometry import LineString, Point

from src.wind_fetch.CoastlineSpatialIndex import CoastlineSpatialIndex
from src.wind_fetch.WindFetchConfig import WindFetchConfig
from src.wind_fetch.geometry_utils import (
    geodesic_forward_point,
    is_in_land_sector,
    normalize_azimuths,
)
from src.wind_fetch.models import MultiDirectionFetchResult

_GEOD = Geod(ellps="WGS84")

# ── Глобальные переменные воркера (инициализируются через initializer) ────────
_WORKER_INDEX: CoastlineSpatialIndex | None = None
_WORKER_CONFIG: WindFetchConfig | None = None
_WORKER_COASTLINE_GDF: gpd.GeoDataFrame | None = None


def _worker_initializer(
    coastline_records: list[dict],
    config: WindFetchConfig,
) -> None:
    global _WORKER_INDEX, _WORKER_CONFIG, _WORKER_COASTLINE_GDF
    _WORKER_CONFIG = config
    _WORKER_COASTLINE_GDF = gpd.GeoDataFrame.from_records(coastline_records)
    _WORKER_COASTLINE_GDF = _WORKER_COASTLINE_GDF.set_geometry("geometry").set_crs("EPSG:4326")
    _WORKER_INDEX = CoastlineSpatialIndex(_WORKER_COASTLINE_GDF)


def _process_chunk(
    chunk: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Обрабатывает порцию точек; вызывается в отдельном процессе."""
    assert _WORKER_CONFIG is not None
    assert _WORKER_INDEX is not None

    cfg = _WORKER_CONFIG
    step_m = cfg.geodesic_step_m
    max_segments = cfg.max_segments_per_ray
    half_sector = cfg.half_land_sector_deg

    azimuths = normalize_azimuths(
        [i * 360.0 / cfg.n_directions for i in range(cfg.n_directions)]
    )

    out: list[dict[str, Any]] = []

    for point_data in chunk:
        source_lon = float(point_data["lon"])
        source_lat = float(point_data["lat"])
        point_id = int(point_data["point_id"])
        normal_az = float(point_data.get("normal_azimuth_deg", 0.0))

        for dir_id, az in enumerate(azimuths):
            skipped = is_in_land_sector(az, normal_az, half_sector)

            if skipped:
                out.append({
                    "point_id": point_id,
                    "direction_id": dir_id,
                    "normal_azimuth_deg": normal_az,
                    "azimuth_deg": az,
                    "source_point_lon": source_lon,
                    "source_point_lat": source_lat,
                    "start_point_lon": source_lon,
                    "start_point_lat": source_lat,
                    "fetch_length_m": cfg.offset_m,
                    "hit_found": False,
                    "hit_lon": None,
                    "hit_lat": None,
                    "used_default_value": False,
                    "skipped_by_land_sector": True,
                })
                continue

            start_lon, start_lat = geodesic_forward_point(
                source_lon, source_lat, az, cfg.offset_m
            )

            hit_lon: float | None = None
            hit_lat: float | None = None
            hit_found = False
            used_default = False

            cur_lon, cur_lat = start_lon, start_lat
            for _ in range(max_segments):
                next_lon, next_lat = geodesic_forward_point(
                    cur_lon, cur_lat, az, step_m
                )
                segment = LineString([(cur_lon, cur_lat), (next_lon, next_lat)])
                for geom in _WORKER_INDEX.query(segment):
                    inter = segment.intersection(geom)
                    if not inter.is_empty:
                        if isinstance(inter, Point):
                            hit_lon, hit_lat = inter.x, inter.y
                        else:
                            pt = inter.geoms[0] if hasattr(inter, "geoms") else inter
                            hit_lon, hit_lat = pt.x, pt.y
                        hit_found = True
                        break
                if hit_found:
                    break
                cur_lon, cur_lat = next_lon, next_lat
            else:
                used_default = True

            if hit_found and hit_lon is not None:
                _, _, fetch_length_m = _GEOD.inv(
                    source_lon, source_lat, hit_lon, hit_lat
                )
            else:
                fetch_length_m = step_m * max_segments

            out.append({
                "point_id": point_id,
                "direction_id": dir_id,
                "normal_azimuth_deg": normal_az,
                "azimuth_deg": az,
                "source_point_lon": source_lon,
                "source_point_lat": source_lat,
                "start_point_lon": start_lon,
                "start_point_lat": start_lat,
                "fetch_length_m": fetch_length_m,
                "hit_found": hit_found,
                "hit_lon": hit_lon,
                "hit_lat": hit_lat,
                "used_default_value": used_default,
                "skipped_by_land_sector": False,
            })

    return out


class WindFetchParallelRunner:
    """Параллельный расчёт многонаправленного wind fetch."""

    def __init__(
        self,
        coastline_paths: list[str | Path],
        points_path: str | Path,
        config: WindFetchConfig | None = None,
    ) -> None:
        self.config = config or WindFetchConfig()

        # ── загрузка береговых линий ─────────────────────────────────────
        coastline_frames: list[gpd.GeoDataFrame] = []
        for p in coastline_paths:
            gdf = gpd.read_file(p)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif str(gdf.crs).upper() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            coastline_frames.append(gdf)

        # ИСПРАВЛЕНО: pd.concat вместо gpd.pd.concat
        combined = pd.concat(coastline_frames, ignore_index=True)
        self._coastline_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

        # ── загрузка точек ───────────────────────────────────────────────
        pts = gpd.read_file(points_path)
        if pts.crs is None:
            pts = pts.set_crs("EPSG:4326")
        elif str(pts.crs).upper() != "EPSG:4326":
            pts = pts.to_crs("EPSG:4326")
        self._points_gdf = pts

    # ── публичный API ────────────────────────────────────────────────────────

    def calculate_multi_direction(self) -> list[MultiDirectionFetchResult]:
        chunks = self._make_chunks()
        coastline_records = self._coastline_gdf.__geo_interface__["features"]

        # Упрощённое представление для передачи между процессами
        coastline_dicts = [
            {"geometry": f["geometry"], **f["properties"]}
            for f in coastline_records
        ]

        all_dicts: list[dict[str, Any]] = []

        with mp.Pool(
            processes=self.config.n_workers,
            initializer=_worker_initializer,
            initargs=(coastline_dicts, self.config),
        ) as pool:
            for chunk_results in pool.imap_unordered(
                _process_chunk, chunks, chunksize=1
            ):
                all_dicts.extend(chunk_results)

        results = [MultiDirectionFetchResult(**d) for d in all_dicts]
        logger.success(f"Параллельный расчёт завершён: {len(results)} результатов")
        return results

    # ── вспомогательные ──────────────────────────────────────────────────────

    def _make_chunks(self) -> list[list[dict[str, Any]]]:
        chunk_size = self.config.chunk_size
        all_points: list[dict[str, Any]] = []
        for _, row in self._points_gdf.iterrows():
            all_points.append({
                "point_id": row.get("point_id", row.name),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "normal_azimuth_deg": float(row.get("normal_azimuth_deg", 0.0)),
            })
        return [
            all_points[i: i + chunk_size]
            for i in range(0, len(all_points), chunk_size)
        ]
