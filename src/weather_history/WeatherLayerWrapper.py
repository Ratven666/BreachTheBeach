from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree


def _slugify(value: Any) -> str:
    text = str(value).strip()
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "unknown"


@dataclass(slots=True)
class WeatherCollection:
    crs: Any
    dates: np.ndarray               # shape (T,)
    speed: np.ndarray               # shape (N, T), float32
    direction: np.ndarray           # shape (N, T), float32
    point_ids: np.ndarray           # shape (N,)
    lat: np.ndarray                 # shape (N,)
    lon: np.ndarray                 # shape (N,)
    req_lat: np.ndarray             # shape (N,)
    req_lon: np.ndarray             # shape (N,)
    ws_unit: str | None
    wd_unit: str | None
    start_date: str | None
    end_date: str | None
    metric_crs: Any
    metric_coords: np.ndarray       # shape (N, 2)
    tree: cKDTree


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

    @classmethod
    def from_file(cls, path: str | Path, working_crs: str | None = None) -> "WeatherLayerWrapper":
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"Weather layer is empty: {path}")
        return cls(gdf, working_crs=working_crs)

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
        assigned["weather_source_point_count"] = len(collection.point_ids)
        assigned["weather_records_count"] = len(collection.dates)

        if output_geojson_path is not None:
            output_geojson_path = Path(output_geojson_path)
            output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
            assigned.to_file(output_geojson_path, driver="GeoJSON", index=False)
            logger.success(f"Assignment GeoJSON written: {output_geojson_path}")

        if output_gpkg_path is not None:
            output_gpkg_path = Path(output_gpkg_path)
            output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
            assigned.to_file(output_gpkg_path, driver="GPKG", layer=output_layer_name, index=False)
            logger.success(f"Assignment GPKG written: {output_gpkg_path}")

        return assigned

    def export_point_files(
        self,
        assigned_gdf: gpd.GeoDataFrame,
        output_dir: str | Path,
        coast_id_column: str | None = "point_id",
        driver: str = "GeoJSON",
    ) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if assigned_gdf.empty:
            logger.warning("Assigned GeoDataFrame is empty; no point files exported.")
            return []

        if coast_id_column and coast_id_column in assigned_gdf.columns:
            id_column = coast_id_column
        elif "point_id" in assigned_gdf.columns:
            id_column = "point_id"
        else:
            assigned_gdf = assigned_gdf.copy()
            assigned_gdf["point_id"] = [f"point_{i:05d}" for i in range(len(assigned_gdf))]
            id_column = "point_id"

        exported_files: list[Path] = []

        for row in assigned_gdf.itertuples(index=False):
            point_id = getattr(row, id_column)
            point_slug = _slugify(point_id)

            point_dir = output_dir / point_slug
            point_dir.mkdir(parents=True, exist_ok=True)

            point_weather_gdf = self._build_point_weather_rows_from_namedtuple(
                row=row,
                point_id=point_id,
                output_crs=assigned_gdf.crs,
            )

            out_path = point_dir / f"{point_slug}.geojson"
            point_weather_gdf.to_file(out_path, driver=driver, index=False)
            exported_files.append(out_path)

        manifest_path = output_dir / "manifest.json"
        manifest = {
            "files_count": len(exported_files),
            "files": [str(path.relative_to(output_dir)) for path in exported_files],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.success(f"Per-point weather files exported: {len(exported_files)}")
        return exported_files

    def export_all_points_weather(
        self,
        assigned_gdf: gpd.GeoDataFrame,
        output_path: str | Path,
        driver: str = "GeoJSON",
        layer_name: str = "all_points_weather",
        coast_id_column: str = "point_id",
    ) -> gpd.GeoDataFrame:
        compact_gdf = self._build_compact_all_points_weather_gdf(
            assigned_gdf=assigned_gdf,
            coast_id_column=coast_id_column,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".gpkg" or driver.upper() == "GPKG":
            compact_gdf.to_file(output_path, driver="GPKG", layer=layer_name, index=False)
        else:
            compact_gdf.to_file(output_path, driver=driver, index=False)

        logger.success(f"All-points compact weather file written: {output_path}")
        return compact_gdf

    def _assign_nearest_vectorized(
            self,
            coastline_gdf: gpd.GeoDataFrame,
            collection: WeatherCollection | None = None,
    ) -> gpd.GeoDataFrame:
        collection = collection or self.collection

        left_metric = coastline_gdf.to_crs(collection.metric_crs)
        left_coords = np.column_stack([left_metric.geometry.x.to_numpy(), left_metric.geometry.y.to_numpy()])

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
        left_coords = np.column_stack([left_metric.geometry.x.to_numpy(), left_metric.geometry.y.to_numpy()])

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

    def _build_weather_collection(
        self,
        weather_gdf: gpd.GeoDataFrame,
        metric_crs: Any,
    ) -> WeatherCollection:
        metric_weather = weather_gdf.to_crs(metric_crs)
        metric_coords = np.column_stack([metric_weather.geometry.x.to_numpy(), metric_weather.geometry.y.to_numpy()])
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

    def _build_point_weather_rows_from_namedtuple(
        self,
        row: Any,
        point_id: Any,
        output_crs: Any = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        row_dict = row._asdict()
        geometry = row_dict["geometry"]
        dates = row_dict.get("dates") or []
        wind_speed = row_dict.get("wind_speed") or []
        wind_dir = row_dict.get("wind_dir") or []

        if not dates:
            empty_row = {
                "point_id": point_id,
                "row_no": 0,
                "date": None,
                "wind_speed": None,
                "wind_direction": None,
                "ws_unit": row_dict.get("ws_unit"),
                "wd_unit": row_dict.get("wd_unit"),
                "weather_strategy": row_dict.get("weather_strategy"),
                "weather_distance_m": row_dict.get("weather_distance_m"),
                "geometry": geometry,
            }
            return gpd.GeoDataFrame([empty_row], geometry="geometry", crs=output_crs)

        records = []
        n = len(dates)

        for i in range(n):
            records.append(
                {
                    "point_id": point_id,
                    "row_no": i,
                    "date": dates[i] if i < len(dates) else None,
                    "wind_speed": float(wind_speed[i]) if i < len(wind_speed) and wind_speed[i] is not None else None,
                    "wind_direction": float(wind_dir[i]) if i < len(wind_dir) and wind_dir[i] is not None else None,
                    "ws_unit": row_dict.get("ws_unit"),
                    "wd_unit": row_dict.get("wd_unit"),
                    "weather_strategy": row_dict.get("weather_strategy"),
                    "weather_distance_m": row_dict.get("weather_distance_m"),
                    "geometry": geometry,
                }
            )

        return gpd.GeoDataFrame(records, geometry="geometry", crs=output_crs)

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
