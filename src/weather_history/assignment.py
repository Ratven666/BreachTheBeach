from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import Point

AssignmentStrategy = Literal["nearest", "idw"]


@dataclass(frozen=True)
class WeatherSeries:
    dates: list[str]
    wind_speed: list[float | None]
    wind_dir: list[float | None]
    ws_unit: str | None
    wd_unit: str | None

    def to_dict(self) -> dict:
        return {
            "dates": self.dates,
            "wind_speed": self.wind_speed,
            "wind_dir": self.wind_dir,
            "ws_unit": self.ws_unit,
            "wd_unit": self.wd_unit,
        }


class WeatherLayerWrapper:
    def __init__(self, weather_gdf: gpd.GeoDataFrame) -> None:
        if weather_gdf.crs is None:
            raise ValueError("Weather GeoDataFrame must have CRS")

        self.weather_gdf = weather_gdf.copy()
        self._restore_weather_columns()

    @classmethod
    def from_file(cls, path: str | Path, layer: str | None = None) -> "WeatherLayerWrapper":
        path = Path(path)
        if layer:
            gdf = gpd.read_file(path, layer=layer)
        else:
            gdf = gpd.read_file(path)
        return cls(gdf)

    def _restore_weather_columns(self) -> None:
        required_columns = ("dates", "wind_speed", "wind_dir")
        for column in required_columns:
            if column not in self.weather_gdf.columns:
                raise ValueError(f"Missing required weather column: {column}")

        self.weather_gdf["dates_list"] = self.weather_gdf["dates"].apply(self._loads_json_list)
        self.weather_gdf["wind_speed_list"] = self.weather_gdf["wind_speed"].apply(self._loads_json_list)
        self.weather_gdf["wind_dir_list"] = self.weather_gdf["wind_dir"].apply(self._loads_json_list)

        lengths_ok = self.weather_gdf.apply(
            lambda row: len(row["dates_list"]) == len(row["wind_speed_list"]) == len(row["wind_dir_list"]),
            axis=1,
        )
        if not bool(lengths_ok.all()):
            raise ValueError("Weather rows contain inconsistent daily array lengths")

    @staticmethod
    def _loads_json_list(value: object) -> list:
        if value is None:
            return []

        if isinstance(value, float) and pd.isna(value):
            return []

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            loaded = json.loads(value)
            if isinstance(loaded, np.ndarray):
                return loaded.tolist()
            if isinstance(loaded, tuple):
                return list(loaded)
            if isinstance(loaded, list):
                return loaded
            raise TypeError(f"JSON value is not a list: {type(loaded)!r}")

        raise TypeError(f"Unsupported JSON list value: {type(value)!r}")

    @staticmethod
    def _ensure_json_text(value: object) -> str:
        normalized = WeatherLayerWrapper._loads_json_list(value)
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _normalize_number(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    def get_series(self, row_index: int) -> WeatherSeries:
        row = self.weather_gdf.iloc[row_index]
        return WeatherSeries(
            dates=[str(v) for v in row["dates_list"]],
            wind_speed=[self._normalize_number(v) for v in row["wind_speed_list"]],
            wind_dir=[self._normalize_number(v) for v in row["wind_dir_list"]],
            ws_unit=row.get("ws_unit"),
            wd_unit=row.get("wd_unit"),
        )

    def assign_to_points(
        self,
        coastline_points_path: str | Path,
        strategy: AssignmentStrategy = "nearest",
        output_geojson_path: str | Path | None = None,
        output_gpkg_path: str | Path | None = None,
        output_layer_name: str = "coastline_weather_points",
        idw_power: float = 2.0,
        idw_k: int = 4,
        working_crs: str = "EPSG:32637",
    ) -> gpd.GeoDataFrame:
        coastline_points = gpd.read_file(coastline_points_path)
        if coastline_points.empty:
            raise ValueError("Coastline points layer is empty")
        if coastline_points.crs is None:
            raise ValueError("Coastline points layer must have CRS")

        coast_wgs84 = coastline_points.to_crs("EPSG:4326")
        weather_wgs84 = self.weather_gdf.to_crs("EPSG:4326")

        coast_metric = coast_wgs84.to_crs(working_crs)
        weather_metric = weather_wgs84.to_crs(working_crs)

        if strategy == "nearest":
            result_gdf = self._assign_nearest(
                coast_metric=coast_metric,
                weather_metric=weather_metric,
            )
        elif strategy == "idw":
            result_gdf = self._assign_idw(
                coast_metric=coast_metric,
                weather_metric=weather_metric,
                idw_power=idw_power,
                idw_k=idw_k,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        result_gdf = result_gdf.to_crs("EPSG:4326")

        if output_geojson_path is not None:
            output_geojson_path = Path(output_geojson_path)
            output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
            result_gdf.to_file(output_geojson_path, driver="GeoJSON")
            logger.success(f"Assigned coastline weather GeoJSON saved: {output_geojson_path}")

        if output_gpkg_path is not None:
            output_gpkg_path = Path(output_gpkg_path)
            output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
            result_gdf.to_file(output_gpkg_path, layer=output_layer_name, driver="GPKG")
            logger.success(
                f"Assigned coastline weather GPKG saved: {output_gpkg_path} layer={output_layer_name}"
            )

        return result_gdf

    def _assign_nearest(
        self,
        coast_metric: gpd.GeoDataFrame,
        weather_metric: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        weather_join = weather_metric[[
            "point_id",
            "lat",
            "lon",
            "dates",
            "wind_speed",
            "wind_dir",
            "ws_unit",
            "wd_unit",
            "geometry",
        ]].copy()

        weather_join["dates"] = weather_join["dates"].apply(self._ensure_json_text)
        weather_join["wind_speed"] = weather_join["wind_speed"].apply(self._ensure_json_text)
        weather_join["wind_dir"] = weather_join["wind_dir"].apply(self._ensure_json_text)

        joined = gpd.sjoin_nearest(
            coast_metric,
            weather_join,
            how="left",
            distance_col="weather_dist_m",
        )

        joined = joined.drop(columns=["index_right"], errors="ignore")
        joined["weather_strategy"] = "nearest"
        joined["src_point_ids"] = joined["point_id"].apply(lambda value: json.dumps([int(value)]))
        joined["src_lat"] = joined["lat"]
        joined["src_lon"] = joined["lon"]
        return joined

    def _assign_idw(
        self,
        coast_metric: gpd.GeoDataFrame,
        weather_metric: gpd.GeoDataFrame,
        idw_power: float,
        idw_k: int,
    ) -> gpd.GeoDataFrame:
        rows: list[dict] = []

        weather_metric = weather_metric.reset_index(drop=True).copy()
        weather_metric["dates_norm"] = weather_metric["dates"].apply(self._loads_json_list)
        weather_metric["wind_speed_norm"] = weather_metric["wind_speed"].apply(self._loads_json_list)
        weather_metric["wind_dir_norm"] = weather_metric["wind_dir"].apply(self._loads_json_list)

        weather_points_xy = [(geom.x, geom.y) for geom in weather_metric.geometry]

        for _, coast_row in coast_metric.iterrows():
            coast_geom: Point = coast_row.geometry
            cx, cy = coast_geom.x, coast_geom.y

            distances: list[tuple[int, float]] = []
            for weather_idx, (wx, wy) in enumerate(weather_points_xy):
                dist = math.hypot(cx - wx, cy - wy)
                distances.append((weather_idx, dist))

            distances.sort(key=lambda item: item[1])
            nearest = distances[:max(1, idw_k)]

            if nearest[0][1] == 0:
                ref_row = weather_metric.iloc[nearest[0][0]]
                out_row = dict(coast_row.drop(labels="geometry"))
                out_row.update({
                    "weather_strategy": "idw",
                    "weather_dist_m": 0.0,
                    "src_point_ids": json.dumps([int(ref_row["point_id"])]),
                    "dates": json.dumps(ref_row["dates_norm"], ensure_ascii=False, separators=(",", ":")),
                    "wind_speed": json.dumps(ref_row["wind_speed_norm"], ensure_ascii=False, separators=(",", ":")),
                    "wind_dir": json.dumps(ref_row["wind_dir_norm"], ensure_ascii=False, separators=(",", ":")),
                    "ws_unit": ref_row["ws_unit"],
                    "wd_unit": ref_row["wd_unit"],
                    "src_lat": ref_row["lat"],
                    "src_lon": ref_row["lon"],
                    "geometry": coast_geom,
                })
                rows.append(out_row)
                continue

            used_rows = [weather_metric.iloc[idx] for idx, _ in nearest]
            used_dists = [dist for _, dist in nearest]

            dates_ref = [str(v) for v in used_rows[0]["dates_norm"]]
            speed_arrays = [
                [self._normalize_number(v) for v in row["wind_speed_norm"]]
                for row in used_rows
            ]
            dir_arrays = [
                [self._normalize_number(v) for v in row["wind_dir_norm"]]
                for row in used_rows
            ]

            if not all([str(v) for v in row["dates_norm"]] == dates_ref for row in used_rows):
                raise ValueError("IDW interpolation requires identical date axes across weather points")

            weights = [1.0 / (dist ** idw_power) for dist in used_dists]
            weights_sum = sum(weights)
            norm_weights = [w / weights_sum for w in weights]

            interp_speed = self._interpolate_scalar_series(speed_arrays, norm_weights)
            interp_dir = self._interpolate_direction_series(dir_arrays, norm_weights)

            out_row = dict(coast_row.drop(labels="geometry"))
            out_row.update({
                "weather_strategy": "idw",
                "weather_dist_m": float(used_dists[0]),
                "src_point_ids": json.dumps([int(row["point_id"]) for row in used_rows]),
                "dates": json.dumps(dates_ref, ensure_ascii=False, separators=(",", ":")),
                "wind_speed": json.dumps(interp_speed, ensure_ascii=False, separators=(",", ":")),
                "wind_dir": json.dumps(interp_dir, ensure_ascii=False, separators=(",", ":")),
                "ws_unit": used_rows[0]["ws_unit"],
                "wd_unit": used_rows[0]["wd_unit"],
                "src_lat": None,
                "src_lon": None,
                "geometry": coast_geom,
            })
            rows.append(out_row)

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=coast_metric.crs)

    @staticmethod
    def _interpolate_scalar_series(
        arrays: list[list[float | None]],
        weights: list[float],
    ) -> list[float | None]:
        size = len(arrays[0])
        result: list[float | None] = []

        for i in range(size):
            values = []
            local_weights = []

            for arr, w in zip(arrays, weights, strict=True):
                value = arr[i]
                if value is not None:
                    values.append(float(value))
                    local_weights.append(w)

            if not values:
                result.append(None)
                continue

            wsum = sum(local_weights)
            result.append(sum(v * w for v, w in zip(values, local_weights, strict=True)) / wsum)

        return result

    @staticmethod
    def _interpolate_direction_series(
        arrays: list[list[float | None]],
        weights: list[float],
    ) -> list[float | None]:
        size = len(arrays[0])
        result: list[float | None] = []

        for i in range(size):
            sin_sum = 0.0
            cos_sum = 0.0
            wsum = 0.0

            for arr, w in zip(arrays, weights, strict=True):
                value = arr[i]
                if value is None:
                    continue
                radians = math.radians(float(value))
                sin_sum += math.sin(radians) * w
                cos_sum += math.cos(radians) * w
                wsum += w

            if wsum == 0:
                result.append(None)
                continue

            angle = math.degrees(math.atan2(sin_sum / wsum, cos_sum / wsum))
            if angle < 0:
                angle += 360.0
            result.append(angle)

        return result

    def to_daily_long_table(self) -> pd.DataFrame:
        rows: list[dict] = []

        for _, row in self.weather_gdf.iterrows():
            dates = [str(v) for v in row["dates_list"]]
            speeds = [self._normalize_number(v) for v in row["wind_speed_list"]]
            dirs = [self._normalize_number(v) for v in row["wind_dir_list"]]

            for date_value, speed, direction in zip(dates, speeds, dirs, strict=True):
                rows.append({
                    "point_id": row["point_id"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "date": date_value,
                    "wind_speed": speed,
                    "wind_dir": direction,
                    "ws_unit": row.get("ws_unit"),
                    "wd_unit": row.get("wd_unit"),
                })

        return pd.DataFrame(rows)
