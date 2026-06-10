from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree
from shapely.geometry import Point

from src.weather_history.domain.WeatherCollection import WeatherCollection
from src.weather_history.domain.WeatherPoint import WeatherPoint
from src.weather_history.services.weather_file_exporter import export_gdf, slugify, write_manifest


class WeatherLayerWrapper:
    DATE_CANDIDATES = ("dates", "date", "time")
    WIND_SPEED_CANDIDATES = ("wind_speed", "windspeed", "wind_speed_10m")
    WIND_DIR_CANDIDATES = ("wind_dir", "wind_direction", "winddir", "wind_direction_10m")
    LAT_CANDIDATES = ("lat", "latitude")
    LON_CANDIDATES = ("lon", "longitude")
    REQ_LAT_CANDIDATES = ("req_lat",)
    REQ_LON_CANDIDATES = ("req_lon",)
    WS_UNIT_CANDIDATES = ("ws_unit",)
    WD_UNIT_CANDIDATES = ("wd_unit",)

    def __init__(self, weather_gdf: gpd.GeoDataFrame, working_crs: str | None = None) -> None:
        if weather_gdf.empty:
            raise ValueError("Weather layer is empty")
        if weather_gdf.crs is None:
            raise ValueError("Weather layer has no CRS")

        self.weather_gdf = weather_gdf.copy()
        self.metric_crs = working_crs or self.weather_gdf.estimate_utm_crs()
        if self.metric_crs is None:
            raise ValueError("Failed to determine metric CRS for weather grid")

        self._normalized_weather = self._normalize_weather_grid(self.weather_gdf)
        self.collection = self._build_weather_collection(self._normalized_weather, self.metric_crs)
        self._assigned_gdf: gpd.GeoDataFrame | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        working_crs: str | None = None,
    ) -> "WeatherLayerWrapper":
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"Weather layer is empty: {path}")
        return cls(gdf, working_crs=working_crs)

    @property
    def has_assigned_points(self) -> bool:
        return self._assigned_gdf is not None and not self._assigned_gdf.empty

    def assign_to_points(
        self,
        coastline_points_path: str | Path,
        strategy: str = "nearest",
        output_geojson_path: str | Path | None = None,
        output_gpkg_path: str | Path | None = None,
        output_layer_name: str = "coastline_weather_points",
        idw_power: float = 2.0,
        idw_k: int = 4,
        working_crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        coastline_gdf = gpd.read_file(coastline_points_path)

        if coastline_gdf.empty:
            raise ValueError(f"Coastline points layer is empty: {coastline_points_path}")
        if coastline_gdf.crs is None:
            raise ValueError("Coastline points layer has no CRS")

        result = coastline_gdf.copy()
        if "point_id" not in result.columns:
            result["point_id"] = [f"point_{i:05d}" for i in range(len(result))]

        collection = self.collection
        if working_crs is not None and str(working_crs) != str(self.collection.metric_crs):
            logger.info(f"Rebuilding weather collection for working CRS: {working_crs}")
            collection = self._build_weather_collection(self._normalized_weather, working_crs)

        if strategy == "nearest":
            assigned = self._assign_nearest_vectorized(result, collection=collection)
        elif strategy == "idw":
            assigned = self._assign_idw_vectorized(
                result,
                collection=collection,
                k=idw_k,
                power=idw_power,
            )
        else:
            raise ValueError(f"Unsupported assignment strategy: {strategy}")

        assigned["weather_strategy"] = strategy
        assigned["weather_source_point_count"] = collection.point_count
        assigned["weather_records_count"] = collection.records_count

        self._assigned_gdf = assigned.copy()

        if output_geojson_path is not None:
            export_gdf(assigned, output_geojson_path, driver="GeoJSON")
            logger.success(f"Assignment GeoJSON written: {output_geojson_path}")

        if output_gpkg_path is not None:
            export_gdf(assigned, output_gpkg_path, driver="GPKG", layer_name=output_layer_name)
            logger.success(f"Assignment GPKG written: {output_gpkg_path}")

        return assigned

    def get_assigned_gdf(self) -> gpd.GeoDataFrame:
        if self._assigned_gdf is None:
            raise ValueError("No assigned points available. Call assign_to_points() first.")
        return self._assigned_gdf.copy()

    def get_point(self, point_id: Any) -> WeatherPoint:
        if self._assigned_gdf is None:
            raise ValueError("No assigned points available. Call assign_to_points() first.")

        selected = self._assigned_gdf[self._assigned_gdf["point_id"] == point_id]
        if selected.empty:
            raise KeyError(f"Point not found: {point_id}")

        return self._row_to_weather_point(selected.iloc[0])

    def points(self) -> list[WeatherPoint]:
        return list(iter(self))

    def __iter__(self) -> Iterator[WeatherPoint]:
        if self._assigned_gdf is None:
            raise ValueError("No assigned points available. Call assign_to_points() first.")

        for _, row in self._assigned_gdf.iterrows():
            yield self._row_to_weather_point(row)

    def export_point_files(
        self,
        assigned_gdf: gpd.GeoDataFrame | None,
        output_dir: str | Path,
        coast_id_column: str = "point_id",
        driver: str = "GeoJSON",
    ) -> list[Path]:
        if assigned_gdf is None:
            assigned_gdf = self.get_assigned_gdf()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if assigned_gdf.empty:
            logger.warning("Assigned GeoDataFrame is empty; no point files exported.")
            return []

        exported_files: list[Path] = []

        for row in assigned_gdf.itertuples(index=False):
            row_dict = row._asdict()
            point_id = row_dict.get(coast_id_column)
            point_slug = slugify(point_id)

            point_dir = output_dir / point_slug
            point_dir.mkdir(parents=True, exist_ok=True)

            point_obj = self.get_point(point_id)
            point_gdf = point_obj.to_timeseries_gdf(crs=assigned_gdf.crs)

            out_path = point_dir / f"{point_slug}.geojson"
            export_gdf(point_gdf, out_path, driver=driver)
            exported_files.append(out_path)

        write_manifest(output_dir, exported_files)
        logger.success(f"Per-point weather files exported: {len(exported_files)}")
        return exported_files

    def export_all_points_weather(
        self,
        assigned_gdf: gpd.GeoDataFrame | None,
        output_path: str | Path,
        driver: str = "GeoJSON",
        layer_name: str = "all_points_weather",
        coast_id_column: str = "point_id",
    ) -> gpd.GeoDataFrame:
        if assigned_gdf is None:
            assigned_gdf = self.get_assigned_gdf()

        compact_gdf = self._build_compact_all_points_weather_gdf(
            assigned_gdf=assigned_gdf,
            coast_id_column=coast_id_column,
        )

        export_gdf(compact_gdf, output_path, driver=driver, layer_name=layer_name)
        logger.success(f"All-points compact weather file written: {output_path}")
        return compact_gdf

    def export_point_timeseries(
        self,
        point_id: Any,
        output_path: str | Path,
        driver: str = "GeoJSON",
    ) -> Path:
        point = self.get_point(point_id)
        crs = self._assigned_gdf.crs if self._assigned_gdf is not None else self.weather_gdf.crs
        point_gdf = point.to_timeseries_gdf(crs=crs)
        return export_gdf(point_gdf, output_path, driver=driver)

    def _assign_nearest_vectorized(
        self,
        coastline_gdf: gpd.GeoDataFrame,
        collection: WeatherCollection | None = None,
    ) -> gpd.GeoDataFrame:
        collection = collection or self.collection

        left_metric = coastline_gdf.to_crs(collection.metric_crs)
        left_coords = np.column_stack([
            left_metric.geometry.x.to_numpy(),
            left_metric.geometry.y.to_numpy(),
        ])

        distances, indices = collection.tree.query(left_coords, k=1)
        distances = np.asarray(distances).reshape(-1)
        indices = np.asarray(indices).reshape(-1)

        result = coastline_gdf.copy()
        result["source_grid_point_id"] = collection.point_ids[indices]
        result["source_lat"] = collection.lat[indices]
        result["source_lon"] = collection.lon[indices]
        result["source_req_lat"] = collection.req_lat[indices]
        result["source_req_lon"] = collection.req_lon[indices]
        result["weather_distance_m"] = distances.astype(float)
        result["dates"] = [collection.dates.tolist() for _ in range(len(result))]
        result["wind_speed"] = [collection.speed[i].astype(float).tolist() for i in indices]
        result["wind_dir"] = [collection.direction[i].astype(float).tolist() for i in indices]
        result["ws_unit"] = collection.ws_unit
        result["wd_unit"] = collection.wd_unit
        result["start_date"] = collection.start_date
        result["end_date"] = collection.end_date
        return result

    def _assign_idw_vectorized(
        self,
        coastline_gdf: gpd.GeoDataFrame,
        k: int = 4,
        power: float = 2.0,
        collection: WeatherCollection | None = None,
    ) -> gpd.GeoDataFrame:
        collection = collection or self.collection

        left_metric = coastline_gdf.to_crs(collection.metric_crs)
        left_coords = np.column_stack([
            left_metric.geometry.x.to_numpy(),
            left_metric.geometry.y.to_numpy(),
        ])

        k_eff = min(max(1, int(k)), len(collection.point_ids))
        distances, indices = collection.tree.query(left_coords, k=k_eff)

        if k_eff == 1:
            distances = np.asarray(distances).reshape(-1, 1)
            indices = np.asarray(indices).reshape(-1, 1)
        else:
            distances = np.asarray(distances)
            indices = np.asarray(indices)

        exact_mask = np.min(distances, axis=1) <= 1e-9

        weights = 1.0 / np.maximum(distances, 1e-9) ** float(power)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / weights_sum

        neighbor_speed = collection.speed[indices]
        neighbor_dir = collection.direction[indices]
        w3 = weights[:, :, None]

        interp_speed = np.sum(neighbor_speed * w3, axis=1)

        ang = np.deg2rad(neighbor_dir)
        sin_sum = np.sum(np.sin(ang) * w3, axis=1)
        cos_sum = np.sum(np.cos(ang) * w3, axis=1)
        interp_dir = (np.rad2deg(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0

        nearest_idx = indices[:, 0]
        nearest_dist = distances[:, 0]

        if np.any(exact_mask):
            exact_rows = np.where(exact_mask)[0]
            exact_src = nearest_idx[exact_rows]
            interp_speed[exact_rows, :] = collection.speed[exact_src, :]
            interp_dir[exact_rows, :] = collection.direction[exact_src, :]
            nearest_dist[exact_rows] = 0.0

        result = coastline_gdf.copy()
        result["source_grid_point_id"] = collection.point_ids[nearest_idx]
        result["source_lat"] = collection.lat[nearest_idx]
        result["source_lon"] = collection.lon[nearest_idx]
        result["source_req_lat"] = collection.req_lat[nearest_idx]
        result["source_req_lon"] = collection.req_lon[nearest_idx]
        result["weather_distance_m"] = nearest_dist.astype(float)
        result["idw_k"] = int(k_eff)
        result["idw_power"] = float(power)
        result["dates"] = [collection.dates.tolist() for _ in range(len(result))]
        result["wind_speed"] = [row.astype(float).tolist() for row in interp_speed]
        result["wind_dir"] = [row.astype(float).tolist() for row in interp_dir]
        result["ws_unit"] = collection.ws_unit
        result["wd_unit"] = collection.wd_unit
        result["start_date"] = collection.start_date
        result["end_date"] = collection.end_date
        return result

    def _row_to_weather_point(self, row: pd.Series) -> WeatherPoint:
        dates = tuple(row.get("dates") or [])
        wind_speed = tuple(row.get("wind_speed") or [])
        wind_dir = tuple(row.get("wind_dir") or [])

        geometry = row.geometry
        if not isinstance(geometry, Point):
            raise TypeError("Assigned row geometry must be Point")

        return WeatherPoint(
            point_id=row.get("point_id"),
            geometry=geometry,
            weather_strategy=row.get("weather_strategy"),
            weather_distance_m=self._safe_float(row.get("weather_distance_m")),
            source_grid_point_id=row.get("source_grid_point_id"),
            source_lat=self._safe_float(row.get("source_lat")),
            source_lon=self._safe_float(row.get("source_lon")),
            source_req_lat=self._safe_float(row.get("source_req_lat")),
            source_req_lon=self._safe_float(row.get("source_req_lon")),
            dates=dates,
            wind_speed=tuple(self._safe_float(v) for v in wind_speed),
            wind_dir=tuple(self._safe_float(v) for v in wind_dir),
            ws_unit=row.get("ws_unit"),
            wd_unit=row.get("wd_unit"),
            start_date=row.get("start_date"),
            end_date=row.get("end_date"),
        )

    def _build_weather_collection(
        self,
        weather_gdf: gpd.GeoDataFrame,
        metric_crs: Any,
    ) -> WeatherCollection:
        metric_weather = weather_gdf.to_crs(metric_crs)
        metric_coords = np.column_stack([
            metric_weather.geometry.x.to_numpy(),
            metric_weather.geometry.y.to_numpy(),
        ])
        tree = cKDTree(metric_coords)

        dates = np.asarray(weather_gdf.iloc[0]["dates"], dtype=object)
        speed = np.vstack(weather_gdf["wind_speed"].to_list()).astype(np.float32)
        direction = np.vstack(weather_gdf["wind_dir"].to_list()).astype(np.float32)

        return WeatherCollection(
            crs=weather_gdf.crs,
            dates=dates,
            speed=speed,
            direction=direction,
            point_ids=weather_gdf["grid_point_id"].to_numpy(dtype=object),
            lat=weather_gdf["lat"].to_numpy(dtype=np.float64),
            lon=weather_gdf["lon"].to_numpy(dtype=np.float64),
            req_lat=weather_gdf["req_lat"].to_numpy(dtype=np.float64),
            req_lon=weather_gdf["req_lon"].to_numpy(dtype=np.float64),
            ws_unit=weather_gdf.iloc[0].get("ws_unit"),
            wd_unit=weather_gdf.iloc[0].get("wd_unit"),
            start_date=weather_gdf.iloc[0].get("start_date"),
            end_date=weather_gdf.iloc[0].get("end_date"),
            metric_crs=metric_crs,
            metric_coords=metric_coords,
            tree=tree,
        )

    def _normalize_weather_grid(self, weather_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        rows: list[dict[str, Any]] = []

        for idx, row in weather_gdf.iterrows():
            dates = self._extract_list_field(row, self.DATE_CANDIDATES)
            wind_speed = self._extract_list_field(row, self.WIND_SPEED_CANDIDATES)
            wind_dir = self._extract_list_field(row, self.WIND_DIR_CANDIDATES)

            if not dates:
                raise ValueError(f"Weather row {idx} does not contain dates")
            if not wind_speed:
                raise ValueError(f"Weather row {idx} does not contain wind_speed")
            if not wind_dir:
                raise ValueError(f"Weather row {idx} does not contain wind_dir")
            if not (len(dates) == len(wind_speed) == len(wind_dir)):
                raise ValueError(
                    f"Weather row {idx} has inconsistent array lengths: "
                    f"dates={len(dates)}, wind_speed={len(wind_speed)}, wind_dir={len(wind_dir)}"
                )

            normalized = row.drop(labels="geometry").to_dict()
            normalized["geometry"] = row.geometry
            normalized["grid_point_id"] = row.get("point_id", idx)
            normalized["lat"] = self._extract_scalar_field(row, self.LAT_CANDIDATES)
            normalized["lon"] = self._extract_scalar_field(row, self.LON_CANDIDATES)
            normalized["req_lat"] = self._extract_scalar_field(row, self.REQ_LAT_CANDIDATES)
            normalized["req_lon"] = self._extract_scalar_field(row, self.REQ_LON_CANDIDATES)
            normalized["dates"] = [self._to_date_string(v) for v in dates]
            normalized["wind_speed"] = [self._safe_float(v) for v in wind_speed]
            normalized["wind_dir"] = [self._normalize_angle(v) for v in wind_dir]
            normalized["ws_unit"] = self._extract_scalar_field(row, self.WS_UNIT_CANDIDATES)
            normalized["wd_unit"] = self._extract_scalar_field(row, self.WD_UNIT_CANDIDATES)
            normalized["start_date"] = row.get("start_date")
            normalized["end_date"] = row.get("end_date")
            rows.append(normalized)

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=weather_gdf.crs)

    def _build_compact_all_points_weather_gdf(
        self,
        assigned_gdf: gpd.GeoDataFrame,
        coast_id_column: str = "point_id",
    ) -> gpd.GeoDataFrame:
        rows: list[dict[str, Any]] = []

        for row in assigned_gdf.itertuples(index=False):
            row_dict = row._asdict()
            rows.append(
                {
                    "point_id": row_dict.get(coast_id_column),
                    "source_grid_point_id": row_dict.get("source_grid_point_id"),
                    "weather_strategy": row_dict.get("weather_strategy"),
                    "weather_distance_m": row_dict.get("weather_distance_m"),
                    "idw_k": row_dict.get("idw_k"),
                    "idw_power": row_dict.get("idw_power"),
                    "source_lat": row_dict.get("source_lat"),
                    "source_lon": row_dict.get("source_lon"),
                    "source_req_lat": row_dict.get("source_req_lat"),
                    "source_req_lon": row_dict.get("source_req_lon"),
                    "start_date": row_dict.get("start_date"),
                    "end_date": row_dict.get("end_date"),
                    "dates": row_dict.get("dates"),
                    "wind_speed": row_dict.get("wind_speed"),
                    "wind_dir": row_dict.get("wind_dir"),
                    "ws_unit": row_dict.get("ws_unit"),
                    "wd_unit": row_dict.get("wd_unit"),
                    "weather_records_count": row_dict.get("weather_records_count"),
                    "geometry": row_dict.get("geometry"),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=assigned_gdf.crs)

    def _extract_list_field(self, row: pd.Series, candidates: tuple[str, ...]) -> list[Any]:
        for name in candidates:
            if name in row.index:
                value = self._to_list(row[name])
                if value:
                    return value
        return []

    def _extract_scalar_field(self, row: pd.Series, candidates: tuple[str, ...]) -> Any:
        for name in candidates:
            if name in row.index:
                value = row[name]
                if not self._is_missing(value):
                    return value
        return None

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        return list(value) if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)) else []

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_angle(value: Any) -> float | None:
        if value is None:
            return None
        try:
            angle = float(value) % 360.0
            if angle < 0:
                angle += 360.0
            return angle
        except Exception:
            return None

    @staticmethod
    def _to_date_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None