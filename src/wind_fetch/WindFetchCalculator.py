# src/wind_fetch/WindFetchCalculator.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyproj import Geod
from shapely.geometry import LineString, Point

from src.wind_fetch.CoastlineSpatialIndex import CoastlineSpatialIndex
from src.wind_fetch.WindFetchConfig import WindFetchConfig
from src.wind_fetch.geometry_utils import geodesic_forward_point, normalize_azimuths
from src.wind_fetch.models import MultiDirectionFetchResult, WindFetchResult

_GEOD = Geod(ellps="WGS84")


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный helper: конечная точка луча
# ─────────────────────────────────────────────────────────────────────────────

def _ray_end_lonlat(
    result: WindFetchResult | MultiDirectionFetchResult,
    azimuth_attr: str = "ray_azimuth_deg",
    step_m: float = 1_000.0,
    max_segments: int = 200,
) -> tuple[float, float]:
    """
    Возвращает (lon, lat) конечной точки луча.
    Если попадание найдено — берёт hit_lon/hit_lat,
    иначе вычисляет точку по максимальной длине луча.
    Вызывается ОДИН РАЗ на строку, результат переиспользуется.
    """
    if result.hit_lon is not None and result.hit_lat is not None:
        return float(result.hit_lon), float(result.hit_lat)
    azimuth = getattr(result, azimuth_attr)
    max_dist = step_m * max_segments
    return geodesic_forward_point(
        result.start_point_lon,
        result.start_point_lat,
        azimuth,
        max_dist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Базовый класс
# ─────────────────────────────────────────────────────────────────────────────

class _BaseFetchCalculator:
    """Общая инфраструктура загрузки данных."""

    def __init__(self, config: WindFetchConfig) -> None:
        self.config = config

    # ── загрузка / нормализация CRS ─────────────────────────────────────────

    @staticmethod
    def _load_gdf_4326(path: str | Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf

    def _load_coastline(self, path: str | Path) -> gpd.GeoDataFrame:
        return self._load_gdf_4326(path)

    def _load_points(self, path: str | Path) -> gpd.GeoDataFrame:
        return self._load_gdf_4326(path)


# ─────────────────────────────────────────────────────────────────────────────
# Однонаправленный калькулятор (WindFetchCalculator)
# ─────────────────────────────────────────────────────────────────────────────

class WindFetchCalculator(_BaseFetchCalculator):
    """Трассирует единственный луч от каждой точки по заданному азимуту."""

    def __init__(
        self,
        coastline_path: str | Path,
        points_path: str | Path,
        azimuth_deg: float,
        config: WindFetchConfig | None = None,
    ) -> None:
        super().__init__(config or WindFetchConfig())
        self.azimuth_deg = float(azimuth_deg) % 360.0
        self._coastline_gdf = self._load_coastline(coastline_path)
        self._points_gdf = self._load_points(points_path)
        self._index = CoastlineSpatialIndex(self._coastline_gdf)

    # ── расчёт ──────────────────────────────────────────────────────────────

    def calculate(self) -> list[WindFetchResult]:
        results: list[WindFetchResult] = []
        for _, row in self._points_gdf.iterrows():
            result = self._trace_ray(row)
            results.append(result)
        return results

    def _trace_ray(self, row: pd.Series) -> WindFetchResult:
        source_lon = float(row.geometry.x)
        source_lat = float(row.geometry.y)
        point_id = int(row.get("point_id", row.name))

        start_lon, start_lat = geodesic_forward_point(
            source_lon, source_lat, self.azimuth_deg, self.config.offset_m
        )

        hit_lon: float | None = None
        hit_lat: float | None = None
        hit_found = False
        used_default = False

        for seg in range(self.config.max_segments_per_ray):
            next_lon, next_lat = geodesic_forward_point(
                start_lon, start_lat, self.azimuth_deg, self.config.geodesic_step_m
            )
            segment = LineString([(start_lon, start_lat), (next_lon, next_lat)])
            candidates = self._index.query(segment)
            for geom in candidates:
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
            start_lon, start_lat = next_lon, next_lat
        else:
            used_default = True

        fetch_length_m: float
        if hit_found and hit_lon is not None and hit_lat is not None:
            _, _, fetch_length_m = _GEOD.inv(
                row.geometry.x, row.geometry.y, hit_lon, hit_lat
            )
        else:
            fetch_length_m = self.config.geodesic_step_m * self.config.max_segments_per_ray

        return WindFetchResult(
            point_id=point_id,
            source_point_lon=source_lon,
            source_point_lat=source_lat,
            start_point_lon=float(row.geometry.x),
            start_point_lat=float(row.geometry.y),
            ray_azimuth_deg=self.azimuth_deg,
            normal_azimuth_deg=float(row.get("normal_azimuth_deg", self.azimuth_deg)),
            fetch_length_m=fetch_length_m,
            hit_found=hit_found,
            hit_lon=hit_lon,
            hit_lat=hit_lat,
            used_default_value=used_default,
        )

    # ── экспорт ─────────────────────────────────────────────────────────────

    def to_rays_geodataframe(self, results: list[WindFetchResult]) -> gpd.GeoDataFrame:
        rows: list[dict[str, Any]] = []
        for r in results:
            end_lon, end_lat = _ray_end_lonlat(
                r,
                azimuth_attr="ray_azimuth_deg",
                step_m=self.config.geodesic_step_m,
                max_segments=self.config.max_segments_per_ray,
            )
            rows.append({
                "point_id": r.point_id,
                "ray_azimuth_deg": r.ray_azimuth_deg,
                "normal_azimuth_deg": r.normal_azimuth_deg,
                "fetch_length_m": r.fetch_length_m,
                "hit_found": r.hit_found,
                "used_default_value": r.used_default_value,
                "end_lon": end_lon,
                "end_lat": end_lat,
                "geometry": LineString([
                    (r.start_point_lon, r.start_point_lat),
                    (end_lon, end_lat),
                ]),
            })
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def save(
        self,
        results: list[WindFetchResult],
        out_dir: str | Path,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / self.config.output_csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            if not results:
                return
            writer = csv.DictWriter(
                fh,
                fieldnames=[f.name for f in WindFetchResult.__dataclass_fields__.values()],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    f: getattr(r, f)
                    for f in WindFetchResult.__dataclass_fields__
                })
        logger.success(f"CSV сохранён: {csv_path}")

        rays_gdf = self.to_rays_geodataframe(results)
        geojson_path = out_dir / self.config.output_geojson_name
        rays_gdf.to_file(geojson_path, driver="GeoJSON")
        logger.success(f"GeoJSON сохранён: {geojson_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Многонаправленный калькулятор (MultiDirectionFetchCalculator)
# ─────────────────────────────────────────────────────────────────────────────

class MultiDirectionFetchCalculator(_BaseFetchCalculator):
    """
    Трассирует N равномерно распределённых лучей от каждой точки.
    Направления в сухопутном секторе пропускаются.
    """

    def __init__(
        self,
        coastline_path: str | Path,
        points_path: str | Path,
        config: WindFetchConfig | None = None,
    ) -> None:
        super().__init__(config or WindFetchConfig())
        self._coastline_gdf = self._load_coastline(coastline_path)
        self._points_gdf = self._load_points(points_path)
        self._index = CoastlineSpatialIndex(self._coastline_gdf)

    def _make_azimuths(self) -> list[float]:
        step = 360.0 / self.config.n_directions
        raw = [i * step for i in range(self.config.n_directions)]
        return normalize_azimuths(raw)

    def calculate(self) -> list[MultiDirectionFetchResult]:
        azimuths = self._make_azimuths()
        results: list[MultiDirectionFetchResult] = []
        for _, row in self._points_gdf.iterrows():
            normal_az = float(row.get("normal_azimuth_deg", 0.0))
            for dir_id, az in enumerate(azimuths):
                r = self._trace_one(row, dir_id, az, normal_az)
                results.append(r)
        return results

    def _trace_one(
        self,
        row: pd.Series,
        direction_id: int,
        azimuth_deg: float,
        normal_azimuth_deg: float,
    ) -> MultiDirectionFetchResult:
        from src.wind_fetch.geometry_utils import is_in_land_sector

        source_lon = float(row.geometry.x)
        source_lat = float(row.geometry.y)
        point_id = int(row.get("point_id", row.name))

        skipped = is_in_land_sector(
            azimuth_deg, normal_azimuth_deg, self.config.half_land_sector_deg
        )

        if skipped:
            return MultiDirectionFetchResult(
                point_id=point_id,
                direction_id=direction_id,
                normal_azimuth_deg=normal_azimuth_deg,
                azimuth_deg=azimuth_deg,
                source_point_lon=source_lon,
                source_point_lat=source_lat,
                start_point_lon=source_lon,
                start_point_lat=source_lat,
                fetch_length_m=self.config.offset_m,
                hit_found=False,
                hit_lon=None,
                hit_lat=None,
                used_default_value=False,
                skipped_by_land_sector=True,
            )

        start_lon, start_lat = geodesic_forward_point(
            source_lon, source_lat, azimuth_deg, self.config.offset_m
        )

        hit_lon: float | None = None
        hit_lat: float | None = None
        hit_found = False
        used_default = False

        cur_lon, cur_lat = start_lon, start_lat
        for _ in range(self.config.max_segments_per_ray):
            next_lon, next_lat = geodesic_forward_point(
                cur_lon, cur_lat, azimuth_deg, self.config.geodesic_step_m
            )
            segment = LineString([(cur_lon, cur_lat), (next_lon, next_lat)])
            for geom in self._index.query(segment):
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
            _, _, fetch_length_m = _GEOD.inv(source_lon, source_lat, hit_lon, hit_lat)
        else:
            fetch_length_m = self.config.geodesic_step_m * self.config.max_segments_per_ray

        return MultiDirectionFetchResult(
            point_id=point_id,
            direction_id=direction_id,
            normal_azimuth_deg=normal_azimuth_deg,
            azimuth_deg=azimuth_deg,
            source_point_lon=source_lon,
            source_point_lat=source_lat,
            start_point_lon=start_lon,
            start_point_lat=start_lat,
            fetch_length_m=fetch_length_m,
            hit_found=hit_found,
            hit_lon=hit_lon,
            hit_lat=hit_lat,
            used_default_value=used_default,
            skipped_by_land_sector=False,
        )

    def to_rays_geodataframe(
        self, results: list[MultiDirectionFetchResult]
    ) -> gpd.GeoDataFrame:
        rows: list[dict[str, Any]] = []
        for r in results:
            end_lon, end_lat = _ray_end_lonlat(
                r,
                azimuth_attr="azimuth_deg",
                step_m=self.config.geodesic_step_m,
                max_segments=self.config.max_segments_per_ray,
            )
            rows.append({
                "point_id": r.point_id,
                "direction_id": r.direction_id,
                "normal_azimuth_deg": r.normal_azimuth_deg,
                "azimuth_deg": r.azimuth_deg,
                "fetch_length_m": r.fetch_length_m,
                "hit_found": r.hit_found,
                "skipped_by_land_sector": r.skipped_by_land_sector,
                "used_default_value": r.used_default_value,
                "end_lon": end_lon,
                "end_lat": end_lat,
                "geometry": LineString([
                    (r.start_point_lon, r.start_point_lat),
                    (end_lon, end_lat),
                ]),
            })
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
