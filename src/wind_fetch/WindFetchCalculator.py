from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Iterable, Sequence

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
from .models import MultiDirectionFetchResult, WindFetchPaths, WindFetchResult


class _BaseFetchCalculator(ABC):
    def __init__(
        self,
        paths: WindFetchPaths,
        config: WindFetchConfig | None = None,
        *,
        dataset_name: str,
    ) -> None:
        self.paths = paths
        self.config = config or WindFetchConfig()

        self.dataset = CoastlineDataset.from_geojson(
            main_path=paths.main_coastline_path,
            other_path=paths.other_coastline_path,
            name=dataset_name,
        )

        self.points_gdf = self._load_points(paths.points_with_normals_path)
        self.coastline_gdf = self._load_coastline()
        self.index = CoastlineSpatialIndex(self.coastline_gdf)

        self._point_geometries = self.points_gdf.geometry.values

        logger.info(
            f"{self.__class__.__name__} initialized: "
            f"coastline_features={len(self.coastline_gdf)}, "
            f"points={len(self.points_gdf)}"
        )

    def _load_points(self, path: str | Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        return gdf

    def _load_coastline(self) -> gpd.GeoDataFrame:
        gdf = self.dataset.combined_gdf

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        return gdf

    @staticmethod
    def _normalize_azimuth_deg(value: float) -> float:
        return float(value) % 360.0

    def _build_start_point(
        self,
        source_lon: float,
        source_lat: float,
        azimuth_deg: float,
        offset_m: float,
    ) -> tuple[float, float]:
        return geodesic_forward_point(
            source_lon,
            source_lat,
            azimuth_deg,
            offset_m,
        )

    def _build_ray(
        self,
        start_lon: float,
        start_lat: float,
        azimuth_deg: float,
    ) -> LineString:
        return build_geodesic_linestring(
            lon=start_lon,
            lat=start_lat,
            azimuth_deg=azimuth_deg,
            total_length_m=self.config.default_fetch_m,
            step_m=self.config.geodesic_step_m,
            max_segments=self.config.max_segments_per_ray,
        )

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


class WindFetchCalculator(_BaseFetchCalculator):
    """
    Вычисляет длину трассы от береговой точки до первого пересечения с береговой линией
    по одному азимуту, заданному в атрибутах точки.

    Азимут трактуется как обычный bearing:
    - 0 = север
    - 90 = восток
    - 180 = юг
    - 270 = запад
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
        super().__init__(
            paths=paths,
            config=config,
            dataset_name="wind_fetch_coastline",
        )
        self.azimuth_field = self._detect_azimuth_field()

        logger.info(
            f"{self.__class__.__name__}: azimuth_field={self.azimuth_field}"
        )

    def _detect_azimuth_field(self) -> str:
        for field in self.NORMAL_AZIMUTH_FIELDS:
            if field in self.points_gdf.columns:
                return field

        raise ValueError(
            f"Points file must contain one of azimuth fields: {self.NORMAL_AZIMUTH_FIELDS}"
        )

    def calculate(self, offset_m: float | None = None) -> list[WindFetchResult]:
        offset = self.config.default_offset_m if offset_m is None else float(offset_m)
        azimuth_values = self.points_gdf[self.azimuth_field].to_numpy()

        results: list[WindFetchResult] = []
        append_result = results.append

        for idx, (geom, azimuth_raw) in enumerate(zip(self._point_geometries, azimuth_values, strict=False), start=1):
            if geom is None or geom.is_empty:
                continue

            source_lon = float(geom.x)
            source_lat = float(geom.y)
            azimuth_deg = self._normalize_azimuth_deg(float(azimuth_raw))

            start_lon, start_lat = self._build_start_point(
                source_lon=source_lon,
                source_lat=source_lat,
                azimuth_deg=azimuth_deg,
                offset_m=offset,
            )

            ray = self._build_ray(
                start_lon=start_lon,
                start_lat=start_lat,
                azimuth_deg=azimuth_deg,
            )

            hit_point = self._find_first_intersection(
                ray=ray,
                start_lon=start_lon,
                start_lat=start_lat,
            )

            if hit_point is None:
                append_result(
                    WindFetchResult(
                        point_id=idx,
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
                WindFetchResult(
                    point_id=idx,
                    source_point_lon=source_lon,
                    source_point_lat=source_lat,
                    start_point_lon=start_lon,
                    start_point_lat=start_lat,
                    normal_azimuth_deg=azimuth_deg,
                    ray_azimuth_deg=azimuth_deg,
                    fetch_length_m=fetch_length_m,
                    hit_found=True,
                    hit_lon=hit_lon,
                    hit_lat=hit_lat,
                    used_default_value=False,
                )
            )

        logger.success(f"Calculated wind fetch for {len(results)} points")
        return results

    def to_dataframe(self, results: Sequence[WindFetchResult]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
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
        )

    def to_geodataframe(self, results: Sequence[WindFetchResult]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            (
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
                for r in results
            ),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def to_rays_geodataframe(self, results: Sequence[WindFetchResult]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            (
                {
                    "point_id": r.point_id,
                    "source_lon": r.source_point_lon,
                    "source_lat": r.source_point_lat,
                    "start_lon": r.start_point_lon,
                    "start_lat": r.start_point_lat,
                    "end_lon": (
                        float(r.hit_lon)
                        if r.hit_lon is not None
                        else geodesic_forward_point(
                            r.start_point_lon,
                            r.start_point_lat,
                            r.ray_azimuth_deg,
                            r.fetch_length_m,
                        )[0]
                    ),
                    "end_lat": (
                        float(r.hit_lat)
                        if r.hit_lat is not None
                        else geodesic_forward_point(
                            r.start_point_lon,
                            r.start_point_lat,
                            r.ray_azimuth_deg,
                            r.fetch_length_m,
                        )[1]
                    ),
                    "azimuth_deg": r.ray_azimuth_deg,
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                    "geometry": LineString(
                        [
                            (r.start_point_lon, r.start_point_lat),
                            (
                                float(r.hit_lon)
                                if r.hit_lon is not None
                                else geodesic_forward_point(
                                    r.start_point_lon,
                                    r.start_point_lat,
                                    r.ray_azimuth_deg,
                                    r.fetch_length_m,
                                )[0],
                                float(r.hit_lat)
                                if r.hit_lat is not None
                                else geodesic_forward_point(
                                    r.start_point_lon,
                                    r.start_point_lat,
                                    r.ray_azimuth_deg,
                                    r.fetch_length_m,
                                )[1],
                            ),
                        ]
                    ),
                }
                for r in results
            ),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def save(
        self,
        results: Sequence[WindFetchResult],
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


class MultiDirectionFetchCalculator(_BaseFetchCalculator):
    """
    Считает трассировку для каждой точки по нескольким направлениям.

    По умолчанию направления: 0..359 градусов с шагом 1 градус.
    """

    def __init__(
        self,
        paths: WindFetchPaths,
        config: WindFetchConfig | None = None,
    ) -> None:
        super().__init__(
            paths=paths,
            config=config,
            dataset_name="multi_direction_wind_fetch_coastline",
        )

    @staticmethod
    def _normalize_azimuths(
        azimuths: Iterable[float] | None,
    ) -> list[float]:
        if azimuths is None:
            return [float(v) for v in range(360)]

        normalized = sorted({float(v) % 360.0 for v in azimuths})
        if not normalized:
            raise ValueError("Azimuth list is empty")

        return normalized

    def calculate(
        self,
        azimuths: Iterable[float] | None = None,
        offset_m: float | None = None,
    ) -> list[MultiDirectionFetchResult]:
        azimuth_list = self._normalize_azimuths(azimuths)
        offset = self.config.default_offset_m if offset_m is None else float(offset_m)

        logger.info(
            f"Calculating multi-direction fetch: "
            f"points={len(self.points_gdf)}, directions={len(azimuth_list)}, offset_m={offset}"
        )

        results: list[MultiDirectionFetchResult] = []
        append_result = results.append

        for point_idx, geom in enumerate(self._point_geometries, start=1):
            if geom is None or geom.is_empty:
                continue

            source_lon = float(geom.x)
            source_lat = float(geom.y)

            for direction_id, azimuth_deg in enumerate(azimuth_list, start=1):
                start_lon, start_lat = self._build_start_point(
                    source_lon=source_lon,
                    source_lat=source_lat,
                    azimuth_deg=azimuth_deg,
                    offset_m=offset,
                )

                ray = self._build_ray(
                    start_lon=start_lon,
                    start_lat=start_lat,
                    azimuth_deg=azimuth_deg,
                )

                hit_point = self._find_first_intersection(
                    ray=ray,
                    start_lon=start_lon,
                    start_lat=start_lat,
                )

                if hit_point is None:
                    append_result(
                        MultiDirectionFetchResult(
                            point_id=point_idx,
                            direction_id=direction_id,
                            azimuth_deg=azimuth_deg,
                            source_point_lon=source_lon,
                            source_point_lat=source_lat,
                            start_point_lon=start_lon,
                            start_point_lat=start_lat,
                            fetch_length_m=self.config.default_fetch_m,
                            hit_found=False,
                            hit_lon=None,
                            hit_lat=None,
                            used_default_value=True,
                        )
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
                    MultiDirectionFetchResult(
                        point_id=point_idx,
                        direction_id=direction_id,
                        azimuth_deg=azimuth_deg,
                        source_point_lon=source_lon,
                        source_point_lat=source_lat,
                        start_point_lon=start_lon,
                        start_point_lat=start_lat,
                        fetch_length_m=fetch_length_m,
                        hit_found=True,
                        hit_lon=hit_lon,
                        hit_lat=hit_lat,
                        used_default_value=False,
                    )
                )

        logger.success(f"Calculated multi-direction fetch rays: {len(results)}")
        return results

    def to_dataframe(self, results: Sequence[MultiDirectionFetchResult]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            {
                "point_id": r.point_id,
                "direction_id": r.direction_id,
                "azimuth_deg": r.azimuth_deg,
                "source_lon": r.source_point_lon,
                "source_lat": r.source_point_lat,
                "start_lon": r.start_point_lon,
                "start_lat": r.start_point_lat,
                "fetch_length_m": r.fetch_length_m,
                "hit_found": r.hit_found,
                "hit_lon": r.hit_lon,
                "hit_lat": r.hit_lat,
                "used_default_value": r.used_default_value,
            }
            for r in results
        )

    def to_points_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            (
                {
                    "point_id": r.point_id,
                    "direction_id": r.direction_id,
                    "azimuth_deg": r.azimuth_deg,
                    "source_lon": r.source_point_lon,
                    "source_lat": r.source_point_lat,
                    "start_lon": r.start_point_lon,
                    "start_lat": r.start_point_lat,
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                    "geometry": Point(r.source_point_lon, r.source_point_lat),
                }
                for r in results
            ),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def to_rays_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            (
                {
                    "point_id": r.point_id,
                    "direction_id": r.direction_id,
                    "azimuth_deg": r.azimuth_deg,
                    "source_lon": r.source_point_lon,
                    "source_lat": r.source_point_lat,
                    "start_lon": r.start_point_lon,
                    "start_lat": r.start_point_lat,
                    "end_lon": (
                        float(r.hit_lon)
                        if r.hit_lon is not None
                        else geodesic_forward_point(
                            r.start_point_lon,
                            r.start_point_lat,
                            r.azimuth_deg,
                            r.fetch_length_m,
                        )[0]
                    ),
                    "end_lat": (
                        float(r.hit_lat)
                        if r.hit_lat is not None
                        else geodesic_forward_point(
                            r.start_point_lon,
                            r.start_point_lat,
                            r.azimuth_deg,
                            r.fetch_length_m,
                        )[1]
                    ),
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                    "geometry": LineString(
                        [
                            (r.start_point_lon, r.start_point_lat),
                            (
                                float(r.hit_lon)
                                if r.hit_lon is not None
                                else geodesic_forward_point(
                                    r.start_point_lon,
                                    r.start_point_lat,
                                    r.azimuth_deg,
                                    r.fetch_length_m,
                                )[0],
                                float(r.hit_lat)
                                if r.hit_lat is not None
                                else geodesic_forward_point(
                                    r.start_point_lon,
                                    r.start_point_lat,
                                    r.azimuth_deg,
                                    r.fetch_length_m,
                                )[1],
                            ),
                        ]
                    ),
                }
                for r in results
            ),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def save_combined(
        self,
        results: Sequence[MultiDirectionFetchResult],
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / self.config.multi_output_csv_name
        points_geojson_path = out_dir / self.config.multi_output_points_name
        rays_geojson_path = out_dir / self.config.multi_output_rays_name

        self.to_dataframe(results).to_csv(csv_path, index=False)
        self.to_points_geodataframe(results).to_file(points_geojson_path, driver="GeoJSON")
        self.to_rays_geodataframe(results).to_file(rays_geojson_path, driver="GeoJSON")

        logger.success(
            f"Saved combined multi-direction outputs: csv={csv_path}, "
            f"points={points_geojson_path}, rays={rays_geojson_path}"
        )

        return {
            "csv": str(csv_path),
            "points_geojson": str(points_geojson_path),
            "rays_geojson": str(rays_geojson_path),
        }

    def save_split_by_direction(
        self,
        results: Sequence[MultiDirectionFetchResult],
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        out_dir = Path(output_dir or self.config.output_dir) / self.config.multi_output_split_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        points_gdf = self.to_points_geodataframe(results)
        rays_gdf = self.to_rays_geodataframe(results)

        saved: dict[str, str] = {}

        for azimuth_deg in sorted(points_gdf["azimuth_deg"].unique()):
            az_label = int(round(float(azimuth_deg))) % 360

            points_part = points_gdf[points_gdf["azimuth_deg"] == azimuth_deg].copy()
            rays_part = rays_gdf[rays_gdf["azimuth_deg"] == azimuth_deg].copy()

            points_path = out_dir / f"points_az_{az_label:03d}.geojson"
            rays_path = out_dir / f"rays_az_{az_label:03d}.geojson"

            points_part.to_file(points_path, driver="GeoJSON")
            rays_part.to_file(rays_path, driver="GeoJSON")

            saved[f"points_{az_label:03d}"] = str(points_path)
            saved[f"rays_{az_label:03d}"] = str(rays_path)

        logger.success(
            f"Saved split multi-direction outputs to {out_dir} "
            f"for {len(points_gdf['azimuth_deg'].unique())} directions"
        )
        return saved
