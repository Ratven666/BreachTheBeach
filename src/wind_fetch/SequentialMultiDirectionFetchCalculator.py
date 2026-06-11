from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyproj import CRS
from shapely import intersection, make_valid, set_precision
from shapely.geometry import LineString, Point
from tqdm.auto import tqdm

from src.coastline.domain.CoastlineDataset import CoastlineDataset

from .CoastlineSpatialIndex import CoastlineSpatialIndex
from .WindFetchConfig import WindFetchConfig
from .models import MultiDirectionFetchResult, WindFetchPaths


class SequentialMultiDirectionFetchCalculator:
    """
    Multi-direction fetch calculator.

    Логика:
    - исходная точка лежит на береговой линии;
    - для неё известен азимут нормали;
    - стартовая точка одна, смещена по нормали на offset_m;
    - азимуты задаются как абсолютные bearings от севера;
    - если азимут попадает в сектор [normal+90, normal+270),
      честная трассировка не считается:
        fetch_length_m = offset_m
    - иначе выполняется обычная трассировка до первого пересечения.
    """

    def __init__(
        self,
        paths: WindFetchPaths,
        config: WindFetchConfig | None = None,
        azimuths_deg: Iterable[float] | None = None,
    ) -> None:
        self.paths = paths
        self.config = config or WindFetchConfig()
        self.azimuths_deg = self._normalize_azimuths(
            azimuths_deg if azimuths_deg is not None else self.config.azimuths_deg
        )

        self.dataset = CoastlineDataset.from_geojson(
            main_path=paths.main_coastline_path,
            other_path=paths.other_coastline_path,
            name="sequential_multi_direction_wind_fetch_coastline",
        )

        self.points_gdf_wgs84 = self._load_points(paths.points_with_normals_path)
        self.coastline_gdf_wgs84 = self._load_coastline()

        self._validate_points_schema(self.points_gdf_wgs84)

        self.metric_crs = self._resolve_metric_crs()

        self.points_gdf_metric = self.points_gdf_wgs84.to_crs(self.metric_crs)
        self.coastline_gdf_metric = self.coastline_gdf_wgs84.to_crs(self.metric_crs)

        self.points_gdf_metric = self.points_gdf_metric[
            self.points_gdf_metric.geometry.notna()
            & ~self.points_gdf_metric.geometry.is_empty
        ].copy()

        self.coastline_gdf_metric = self.coastline_gdf_metric[
            self.coastline_gdf_metric.geometry.notna()
            & ~self.coastline_gdf_metric.geometry.is_empty
        ].copy()

        self.coastline_gdf_metric = self._prepare_coastline_geometry(
            self.coastline_gdf_metric
        )

        self.index = CoastlineSpatialIndex(self.coastline_gdf_metric)

        logger.info(
            f"{self.__class__.__name__} initialized: "
            f"points={len(self.points_gdf_metric)}, "
            f"coastline_features={len(self.coastline_gdf_metric)}, "
            f"azimuths={len(self.azimuths_deg)}, "
            f"metric_crs={self.metric_crs}, "
            f"normal_field={self.config.normal_azimuth_field}, "
            f"precision_grid_m={self.config.precision_grid_m}"
        )

    @staticmethod
    def _normalize_azimuth_deg(angle: float) -> float:
        return float(angle) % 360.0

    def _normalize_azimuths(self, azimuths_deg: Iterable[float]) -> list[float]:
        values = sorted({self._normalize_azimuth_deg(v) for v in azimuths_deg})
        if not values:
            raise ValueError("azimuths_deg is empty")
        return values

    @staticmethod
    def _load_gdf_4326(path: str | Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        return gdf

    def _load_points(self, path: str | Path) -> gpd.GeoDataFrame:
        gdf = self._load_gdf_4326(path)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    def _load_coastline(self) -> gpd.GeoDataFrame:
        gdf = self.dataset.combined_gdf

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    def _validate_points_schema(self, gdf: gpd.GeoDataFrame) -> None:
        field = self.config.normal_azimuth_field
        if field not in gdf.columns:
            raise ValueError(
                f"Points file must contain normal azimuth field '{field}'. "
                f"Available columns: {list(gdf.columns)}"
            )

    def _resolve_metric_crs(self) -> CRS:
        metric_crs = self.dataset.metric_crs
        if metric_crs is None:
            raise ValueError("Failed to resolve metric CRS for fetch calculation")
        return CRS.from_user_input(metric_crs)

    def _prepare_geometry(self, geom):
        if geom is None or geom.is_empty:
            return geom

        out = geom

        if self.config.use_make_valid:
            try:
                out = make_valid(out)
            except Exception:
                logger.exception("make_valid failed for geometry")

        try:
            out = set_precision(out, grid_size=self.config.precision_grid_m)
        except Exception:
            logger.exception("set_precision failed for geometry")

        return out

    def _prepare_coastline_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = gdf.copy()
        out["geometry"] = out.geometry.apply(self._prepare_geometry)
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
        return out

    @staticmethod
    def _azimuth_to_dxdy(azimuth_deg: float, length_m: float) -> tuple[float, float]:
        import math

        rad = math.radians(float(azimuth_deg) % 360.0)
        dx = length_m * math.sin(rad)
        dy = length_m * math.cos(rad)
        return dx, dy

    def _project_xy_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        point_metric = gpd.GeoSeries([Point(x, y)], crs=self.metric_crs)
        point_wgs84 = point_metric.to_crs("EPSG:4326")
        geom = point_wgs84.iloc[0]
        return float(geom.x), float(geom.y)

    def _iter_point_rows(self) -> list[tuple[int, pd.Series]]:
        rows: list[tuple[int, pd.Series]] = []

        for point_id, (_, row) in enumerate(self.points_gdf_metric.iterrows(), start=1):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if not isinstance(geom, Point):
                continue
            rows.append((point_id, row))

        return rows

    def _build_shifted_start_point_from_normal(
        self,
        source_x: float,
        source_y: float,
        normal_azimuth_deg: float,
        offset_m: float,
    ) -> tuple[float, float]:
        if offset_m <= 0:
            raise ValueError(f"offset_m must be > 0, got {offset_m}")

        dx, dy = self._azimuth_to_dxdy(normal_azimuth_deg, offset_m)
        return source_x + dx, source_y + dy

    def _build_ray(
        self,
        start_x: float,
        start_y: float,
        azimuth_deg: float,
        fetch_length_m: float,
    ) -> LineString:
        dx, dy = self._azimuth_to_dxdy(azimuth_deg, fetch_length_m)
        ray = LineString([(start_x, start_y), (start_x + dx, start_y + dy)])
        return self._prepare_geometry(ray)

    def _extract_points_from_intersection(self, geom) -> list[Point]:
        if geom is None or geom.is_empty:
            return []

        geom_type = geom.geom_type

        if geom_type == "Point":
            return [geom]

        if geom_type == "MultiPoint":
            return [g for g in geom.geoms if not g.is_empty]

        if geom_type == "LineString":
            coords = list(geom.coords)
            if not coords:
                return []
            return [Point(coords[0]), Point(coords[-1])]

        if geom_type == "MultiLineString":
            pts: list[Point] = []
            for line in geom.geoms:
                if line.is_empty:
                    continue
                coords = list(line.coords)
                if not coords:
                    continue
                pts.append(Point(coords[0]))
                pts.append(Point(coords[-1]))
            return pts

        if hasattr(geom, "geoms"):
            pts: list[Point] = []
            for part in geom.geoms:
                pts.extend(self._extract_points_from_intersection(part))
            return pts

        return []

    def _robust_intersection(self, ray, coastline):
        try:
            return intersection(
                ray,
                coastline,
                grid_size=self.config.precision_grid_m,
            )
        except Exception:
            logger.exception("Robust intersection failed")
            return ray.intersection(coastline)

    def _find_first_intersection(
        self,
        ray: LineString,
        source_x: float,
        source_y: float,
        start_x: float,
        start_y: float,
    ) -> Point | None:
        candidates = self.index.query(ray)
        if not candidates:
            return None

        source_point = Point(source_x, source_y)
        start_point = Point(start_x, start_y)

        nearest_point: Point | None = None
        nearest_dist: float | None = None

        for coastline in candidates:
            if coastline is None or coastline.is_empty:
                continue

            coastline_prepared = self._prepare_geometry(coastline)
            if coastline_prepared is None or coastline_prepared.is_empty:
                continue

            if not coastline_prepared.intersects(ray):
                continue

            inter = self._robust_intersection(ray, coastline_prepared)
            if inter.is_empty:
                continue

            points = self._extract_points_from_intersection(inter)
            if not points:
                continue

            for pt in points:
                pt = self._prepare_geometry(pt)
                if pt is None or pt.is_empty:
                    continue

                dist_from_source = source_point.distance(pt)
                if dist_from_source <= self.config.coastal_exclusion_m:
                    continue

                dist_from_start = start_point.distance(pt)
                if dist_from_start <= self.config.coastal_exclusion_m:
                    continue

                if nearest_dist is None or dist_from_start < nearest_dist:
                    nearest_dist = dist_from_start
                    nearest_point = pt

        return nearest_point

    def _is_in_land_sector(
        self,
        normal_azimuth_deg: float,
        azimuth_deg: float,
    ) -> bool:
        """
        Сухопутный сектор задаётся как [normal+90, normal+270),
        что эквивалентно проверке:
            delta = (azimuth - normal) mod 360
            90 <= delta < 270
        """
        delta = (self._normalize_azimuth_deg(azimuth_deg) - self._normalize_azimuth_deg(normal_azimuth_deg)) % 360.0
        return 90.0 <= delta < 270.0

    def calculate(
        self,
        offset_m: float | None = None,
        show_progress: bool = True,
        log_every_points: int = 25,
    ) -> list[MultiDirectionFetchResult]:
        offset = self.config.default_offset_m if offset_m is None else float(offset_m)

        if offset <= 0:
            raise ValueError(f"offset_m must be > 0, got {offset}")

        point_rows = self._iter_point_rows()
        total_points = len(point_rows)
        total_directions = len(self.azimuths_deg)
        total_rays = total_points * total_directions

        logger.info(
            f"Starting sequential multi-direction fetch: "
            f"points={total_points}, directions={total_directions}, "
            f"total_rays={total_rays}, offset_m={offset}, "
            f"normal_field={self.config.normal_azimuth_field}"
        )

        results: list[MultiDirectionFetchResult] = []
        append_result = results.append

        progress = None
        if show_progress:
            progress = tqdm(
                total=total_rays,
                desc="Wind fetch",
                unit="ray",
                dynamic_ncols=True,
            )

        try:
            for processed_points, (point_id, row) in enumerate(point_rows, start=1):
                source_geom = row.geometry
                source_x = float(source_geom.x)
                source_y = float(source_geom.y)

                normal_azimuth_deg = self._normalize_azimuth_deg(
                    float(row[self.config.normal_azimuth_field])
                )

                start_x, start_y = self._build_shifted_start_point_from_normal(
                    source_x=source_x,
                    source_y=source_y,
                    normal_azimuth_deg=normal_azimuth_deg,
                    offset_m=offset,
                )

                source_lon, source_lat = self._project_xy_to_lonlat(source_x, source_y)
                start_lon, start_lat = self._project_xy_to_lonlat(start_x, start_y)

                for direction_id, azimuth_deg in enumerate(self.azimuths_deg, start=1):
                    skipped_by_land_sector = self._is_in_land_sector(
                        normal_azimuth_deg=normal_azimuth_deg,
                        azimuth_deg=azimuth_deg,
                    )

                    if skipped_by_land_sector:
                        append_result(
                            MultiDirectionFetchResult(
                                point_id=point_id,
                                direction_id=direction_id,
                                normal_azimuth_deg=normal_azimuth_deg,
                                azimuth_deg=azimuth_deg,
                                source_point_lon=source_lon,
                                source_point_lat=source_lat,
                                start_point_lon=start_lon,
                                start_point_lat=start_lat,
                                fetch_length_m=offset,
                                hit_found=False,
                                hit_lon=None,
                                hit_lat=None,
                                used_default_value=True,
                                skipped_by_land_sector=True,
                            )
                        )

                        if progress is not None:
                            progress.update(1)
                        continue

                    ray = self._build_ray(
                        start_x=start_x,
                        start_y=start_y,
                        azimuth_deg=azimuth_deg,
                        fetch_length_m=self.config.default_fetch_m,
                    )

                    hit_point_metric = self._find_first_intersection(
                        ray=ray,
                        source_x=source_x,
                        source_y=source_y,
                        start_x=start_x,
                        start_y=start_y,
                    )

                    if hit_point_metric is None:
                        append_result(
                            MultiDirectionFetchResult(
                                point_id=point_id,
                                direction_id=direction_id,
                                normal_azimuth_deg=normal_azimuth_deg,
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
                                skipped_by_land_sector=False,
                            )
                        )
                    else:
                        hit_x = float(hit_point_metric.x)
                        hit_y = float(hit_point_metric.y)
                        hit_lon, hit_lat = self._project_xy_to_lonlat(hit_x, hit_y)
                        fetch_length_m = Point(start_x, start_y).distance(hit_point_metric)

                        append_result(
                            MultiDirectionFetchResult(
                                point_id=point_id,
                                direction_id=direction_id,
                                normal_azimuth_deg=normal_azimuth_deg,
                                azimuth_deg=azimuth_deg,
                                source_point_lon=source_lon,
                                source_point_lat=source_lat,
                                start_point_lon=start_lon,
                                start_point_lat=start_lat,
                                fetch_length_m=float(fetch_length_m),
                                hit_found=True,
                                hit_lon=hit_lon,
                                hit_lat=hit_lat,
                                used_default_value=False,
                                skipped_by_land_sector=False,
                            )
                        )

                    if progress is not None:
                        progress.update(1)

                if log_every_points > 0 and (
                    processed_points % log_every_points == 0
                    or processed_points == total_points
                ):
                    logger.info(
                        f"Progress by points: {processed_points}/{total_points} "
                        f"({processed_points / total_points * 100:.1f}%)"
                    )

        finally:
            if progress is not None:
                progress.close()

        logger.success(f"Calculated sequential multi-direction fetch rays: {len(results)}")
        return results

    def to_dataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            {
                "point_id": r.point_id,
                "direction_id": r.direction_id,
                "normal_azimuth_deg": r.normal_azimuth_deg,
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
                "skipped_by_land_sector": r.skipped_by_land_sector,
            }
            for r in results
        )

    def to_points_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        seen: set[tuple[int, float, float]] = set()
        rows = []

        for r in results:
            key = (r.point_id, r.source_point_lon, r.source_point_lat)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "point_id": r.point_id,
                    "normal_azimuth_deg": r.normal_azimuth_deg,
                    "geometry": Point(r.source_point_lon, r.source_point_lat),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def to_start_points_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        seen: set[tuple[int, float, float]] = set()
        rows = []

        for r in results:
            key = (r.point_id, r.start_point_lon, r.start_point_lat)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "point_id": r.point_id,
                    "normal_azimuth_deg": r.normal_azimuth_deg,
                    "geometry": Point(r.start_point_lon, r.start_point_lat),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def to_offset_segments_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        seen: set[int] = set()
        rows = []

        for r in results:
            if r.point_id in seen:
                continue
            seen.add(r.point_id)

            rows.append(
                {
                    "point_id": r.point_id,
                    "normal_azimuth_deg": r.normal_azimuth_deg,
                    "geometry": LineString(
                        [
                            (r.source_point_lon, r.source_point_lat),
                            (r.start_point_lon, r.start_point_lat),
                        ]
                    ),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def to_hit_points_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        rows = []

        for r in results:
            if not r.hit_found or r.hit_lon is None or r.hit_lat is None:
                continue

            rows.append(
                {
                    "point_id": r.point_id,
                    "direction_id": r.direction_id,
                    "normal_azimuth_deg": r.normal_azimuth_deg,
                    "azimuth_deg": r.azimuth_deg,
                    "fetch_length_m": r.fetch_length_m,
                    "skipped_by_land_sector": r.skipped_by_land_sector,
                    "geometry": Point(r.hit_lon, r.hit_lat),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def to_rays_geodataframe(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> gpd.GeoDataFrame:
        rows = []

        for r in results:
            if r.hit_lon is not None and r.hit_lat is not None:
                end_lon = float(r.hit_lon)
                end_lat = float(r.hit_lat)
            else:
                start_metric = gpd.GeoSeries(
                    [Point(r.start_point_lon, r.start_point_lat)],
                    crs="EPSG:4326",
                ).to_crs(self.metric_crs).iloc[0]

                start_x = float(start_metric.x)
                start_y = float(start_metric.y)

                dx, dy = self._azimuth_to_dxdy(r.azimuth_deg, r.fetch_length_m)
                end_x = start_x + dx
                end_y = start_y + dy
                end_lon, end_lat = self._project_xy_to_lonlat(end_x, end_y)

            rows.append(
                {
                    "point_id": r.point_id,
                    "direction_id": r.direction_id,
                    "normal_azimuth_deg": r.normal_azimuth_deg,
                    "azimuth_deg": r.azimuth_deg,
                    "fetch_length_m": r.fetch_length_m,
                    "hit_found": r.hit_found,
                    "hit_lon": r.hit_lon,
                    "hit_lat": r.hit_lat,
                    "used_default_value": r.used_default_value,
                    "skipped_by_land_sector": r.skipped_by_land_sector,
                    "geometry": LineString(
                        [
                            (r.start_point_lon, r.start_point_lat),
                            (end_lon, end_lat),
                        ]
                    ),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def save_combined(
        self,
        results: Sequence[MultiDirectionFetchResult],
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / self.config.multi_output_csv_name
        source_points_path = out_dir / self.config.multi_output_points_name
        start_points_path = out_dir / self.config.multi_output_start_points_name
        offset_segments_path = out_dir / self.config.multi_output_offset_segments_name
        rays_path = out_dir / self.config.multi_output_rays_name
        hit_points_path = out_dir / self.config.multi_output_hit_points_name

        self.to_dataframe(results).to_csv(csv_path, index=False)
        self.to_points_geodataframe(results).to_file(source_points_path, driver="GeoJSON")
        self.to_start_points_geodataframe(results).to_file(start_points_path, driver="GeoJSON")
        self.to_offset_segments_geodataframe(results).to_file(offset_segments_path, driver="GeoJSON")
        self.to_rays_geodataframe(results).to_file(rays_path, driver="GeoJSON")
        self.to_hit_points_geodataframe(results).to_file(hit_points_path, driver="GeoJSON")

        logger.success(
            f"Saved combined outputs: "
            f"csv={csv_path}, "
            f"source_points={source_points_path}, "
            f"start_points={start_points_path}, "
            f"offset_segments={offset_segments_path}, "
            f"rays={rays_path}, "
            f"hit_points={hit_points_path}"
        )

        return {
            "csv": str(csv_path),
            "source_points_geojson": str(source_points_path),
            "start_points_geojson": str(start_points_path),
            "offset_segments_geojson": str(offset_segments_path),
            "rays_geojson": str(rays_path),
            "hit_points_geojson": str(hit_points_path),
        }

    def save_split_by_direction(
        self,
        results: Sequence[MultiDirectionFetchResult],
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        out_dir = Path(output_dir or self.config.output_dir) / self.config.multi_output_split_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        rays_gdf = self.to_rays_geodataframe(results)
        hits_gdf = self.to_hit_points_geodataframe(results)

        saved: dict[str, str] = {}

        source_points_path = out_dir / "source_points.geojson"
        start_points_path = out_dir / "start_points.geojson"
        offset_segments_path = out_dir / "offset_segments.geojson"

        self.to_points_geodataframe(results).to_file(source_points_path, driver="GeoJSON")
        self.to_start_points_geodataframe(results).to_file(start_points_path, driver="GeoJSON")
        self.to_offset_segments_geodataframe(results).to_file(offset_segments_path, driver="GeoJSON")

        saved["source_points"] = str(source_points_path)
        saved["start_points"] = str(start_points_path)
        saved["offset_segments"] = str(offset_segments_path)

        for azimuth_deg in sorted(rays_gdf["azimuth_deg"].unique()):
            az_label = int(round(float(azimuth_deg))) % 360

            rays_part = rays_gdf[rays_gdf["azimuth_deg"] == azimuth_deg].copy()
            hits_part = hits_gdf[hits_gdf["azimuth_deg"] == azimuth_deg].copy()

            rays_path = out_dir / f"rays_az_{az_label:03d}.geojson"
            hits_path = out_dir / f"hit_points_az_{az_label:03d}.geojson"

            rays_part.to_file(rays_path, driver="GeoJSON")
            hits_part.to_file(hits_path, driver="GeoJSON")

            saved[f"rays_{az_label:03d}"] = str(rays_path)
            saved[f"hit_points_{az_label:03d}"] = str(hits_path)

        logger.success(
            f"Saved split outputs to {out_dir} "
            f"for {len(rays_gdf['azimuth_deg'].unique())} directions"
        )
        return saved
