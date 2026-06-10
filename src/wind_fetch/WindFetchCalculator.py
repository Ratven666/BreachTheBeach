from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import LineString, Point

from src.coastline.domain.CoastlineDataset import CoastlineDataset

from .CoastlineSpatialIndex import CoastlineSpatialIndex
from .WindFetchConfig import WindFetchConfig
from .geometry_utils import (
    build_geodesic_linestring,
    extract_points_from_intersection,
    geodesic_distance_m,
    geodesic_forward_point,
)
from .WindFetchResult import WindFetchPaths, WindFetchResult


class WindFetchCalculator:
    """
    Вычисляет длину трассы от береговой точки до первого пересечения с береговой линией.

    Ожидается, что файл с точками содержит:
    - geometry (Point, EPSG:4326)
    - поле азимута направления трассировки в градусах

    Поддерживаемые имена поля азимута:
    - normal_azimuth_deg
    - normal_azimuth
    - azimuth_deg
    - azimuth
    - bearing_deg
    - bearing

    ВАЖНО:
    Входной азимут трактуется как обычный геодезический bearing:
    - 0   = север
    - 90  = восток
    - 180 = юг
    - 270 = запад

    То есть значение из файла используется напрямую в pyproj.Geod.fwd
    без разворота на 180 градусов.
    """

    NORMAL_AZIMUTH_FIELDS = (
        "normal_azimuth_deg",
        "normal_azimuth",
        "azimuth_deg",
        "azimuth",
        "bearing_deg",
        "bearing",
    )

    def __init__(
        self,
        paths: WindFetchPaths,
        config: WindFetchConfig | None = None,
    ) -> None:
        self.paths = paths
        self.config = config or WindFetchConfig()

        self.dataset = CoastlineDataset.from_geojson(
            main_path=paths.main_coastline_path,
            other_path=paths.other_coastline_path,
            name="wind_fetch_coastline",
        )

        self.points_gdf = gpd.read_file(paths.points_with_normals_path)
        if self.points_gdf.crs is None:
            self.points_gdf = self.points_gdf.set_crs("EPSG:4326")
        elif str(self.points_gdf.crs) != "EPSG:4326":
            self.points_gdf = self.points_gdf.to_crs("EPSG:4326")

        self.coastline_gdf = self.dataset.combined_gdf
        if self.coastline_gdf.crs is None:
            self.coastline_gdf = self.coastline_gdf.set_crs("EPSG:4326")
        elif str(self.coastline_gdf.crs) != "EPSG:4326":
            self.coastline_gdf = self.coastline_gdf.to_crs("EPSG:4326")

        self.index = CoastlineSpatialIndex(self.coastline_gdf)
        self.azimuth_field = self._detect_azimuth_field()

        logger.info(
            f"WindFetchCalculator initialized: coastline_features={len(self.coastline_gdf)}, "
            f"points={len(self.points_gdf)}, azimuth_field={self.azimuth_field}"
        )

    def _detect_azimuth_field(self) -> str:
        for field in self.NORMAL_AZIMUTH_FIELDS:
            if field in self.points_gdf.columns:
                return field

        raise ValueError(
            f"Points file must contain one of azimuth fields: {self.NORMAL_AZIMUTH_FIELDS}"
        )

    @staticmethod
    def _normalize_azimuth_deg(value: float) -> float:
        return float(value) % 360.0

    def calculate(self, offset_m: float | None = None) -> list[WindFetchResult]:
        offset = self.config.default_offset_m if offset_m is None else float(offset_m)
        results: list[WindFetchResult] = []

        for idx, row in self.points_gdf.iterrows():
            point: Point = row.geometry
            if point is None or point.is_empty:
                continue

            source_lon = float(point.x)
            source_lat = float(point.y)

            azimuth_deg = self._normalize_azimuth_deg(float(row[self.azimuth_field]))

            start_lon, start_lat = geodesic_forward_point(
                source_lon,
                source_lat,
                azimuth_deg,
                offset,
            )

            ray = build_geodesic_linestring(
                lon=start_lon,
                lat=start_lat,
                azimuth_deg=azimuth_deg,
                total_length_m=self.config.default_fetch_m,
                step_m=self.config.geodesic_step_m,
                max_segments=self.config.max_segments_per_ray,
            )

            hit_point = self._find_first_intersection(
                ray=ray,
                start_lon=start_lon,
                start_lat=start_lat,
            )

            if hit_point is None:
                result = WindFetchResult(
                    point_id=int(idx) + 1,
                    source_point_lon=source_lon,
                    source_point_lat=source_lat,
                    start_point_lon=start_lon,
                    start_point_lat=start_lat,
                    normal_azimuth_deg=azimuth_deg,
                    ray_azimuth_deg=azimuth_deg,
                    fetch_length_m=self.config.default_fetch_m,
                    hit_found=False,
                    hit_lon=None,
                    hit_lat=None,
                    used_default_value=True,
                )
            else:
                hit_lon = float(hit_point.x)
                hit_lat = float(hit_point.y)
                distance_m = geodesic_distance_m(
                    start_lon,
                    start_lat,
                    hit_lon,
                    hit_lat,
                )

                result = WindFetchResult(
                    point_id=int(idx) + 1,
                    source_point_lon=source_lon,
                    source_point_lat=source_lat,
                    start_point_lon=start_lon,
                    start_point_lat=start_lat,
                    normal_azimuth_deg=azimuth_deg,
                    ray_azimuth_deg=azimuth_deg,
                    fetch_length_m=distance_m,
                    hit_found=True,
                    hit_lon=hit_lon,
                    hit_lat=hit_lat,
                    used_default_value=False,
                )

            results.append(result)

        logger.success(f"Calculated wind fetch for {len(results)} points")
        return results

    def _find_first_intersection(
        self,
        ray: LineString,
        start_lon: float,
        start_lat: float,
    ) -> Point | None:
        candidates = self.index.query(ray)
        if not candidates:
            return None

        nearest_point: Point | None = None
        nearest_dist: float | None = None

        for coastline in candidates:
            if coastline is None or coastline.is_empty:
                continue

            inter = ray.intersection(coastline)
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

    def to_dataframe(self, results: list[WindFetchResult]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "point_id": r.point_id,
                    "source_point_lon": r.source_point_lon,
                    "source_point_lat": r.source_point_lat,
                    "start_point_lon": r.start_point_lon,
                    "start_point_lat": r.start_point_lat,
                    "azimuth_deg": r.ray_azimuth_deg,
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                }
                for r in results
            ]
        )

    def to_geodataframe(self, results: list[WindFetchResult]) -> gpd.GeoDataFrame:
        rows = []

        for r in results:
            rows.append(
                {
                    "point_id": r.point_id,
                    "source_lon": r.source_point_lon,
                    "source_lat": r.source_point_lat,
                    "start_lon": r.start_point_lon,
                    "start_lat": r.start_point_lat,
                    "azimuth_deg": r.ray_azimuth_deg,
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                    "geometry": Point(r.source_point_lon, r.source_point_lat),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def to_rays_geodataframe(self, results: list[WindFetchResult]) -> gpd.GeoDataFrame:
        rows = []

        for r in results:
            if r.hit_lon is not None and r.hit_lat is not None:
                end_lon = float(r.hit_lon)
                end_lat = float(r.hit_lat)
            else:
                end_lon, end_lat = geodesic_forward_point(
                    r.start_point_lon,
                    r.start_point_lat,
                    r.ray_azimuth_deg,
                    r.fetch_length_m,
                )

            rows.append(
                {
                    "point_id": r.point_id,
                    "source_lon": r.source_point_lon,
                    "source_lat": r.source_point_lat,
                    "start_lon": r.start_point_lon,
                    "start_lat": r.start_point_lat,
                    "end_lon": end_lon,
                    "end_lat": end_lat,
                    "azimuth_deg": r.ray_azimuth_deg,
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                    "geometry": LineString(
                        [
                            (r.start_point_lon, r.start_point_lat),
                            (end_lon, end_lat),
                        ]
                    ),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def save(
        self,
        results: list[WindFetchResult],
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / self.config.output_csv_name
        points_geojson_path = out_dir / self.config.output_geojson_name
        rays_geojson_path = out_dir / f"rays_{self.config.output_geojson_name}"

        self.to_dataframe(results).to_csv(csv_path, index=False)
        self.to_geodataframe(results).to_file(points_geojson_path, driver="GeoJSON")
        self.to_rays_geodataframe(results).to_file(rays_geojson_path, driver="GeoJSON")

        logger.success(
            f"Saved wind fetch outputs: csv={csv_path}, "
            f"points={points_geojson_path}, rays={rays_geojson_path}"
        )

        return {
            "csv": str(csv_path),
            "points_geojson": str(points_geojson_path),
            "rays_geojson": str(rays_geojson_path),
        }