from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger


def _slugify(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


class WeatherLayerWrapper:
    def __init__(self, weather_gdf: gpd.GeoDataFrame) -> None:
        if weather_gdf.crs is None:
            raise ValueError("Weather layer has no CRS")
        self.weather_gdf = weather_gdf

    @classmethod
    def from_file(cls, path: str | Path) -> "WeatherLayerWrapper":
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"Weather layer is empty: {path}")
        return cls(gdf)

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

        weather_index_fields = []
        for field in ("grid_lat", "grid_lon", "latitude", "longitude"):
            if field in self.weather_gdf.columns:
                weather_index_fields.append(field)

        result["weather_strategy"] = strategy
        result["weather_records_count"] = 0
        result["weather_source_point_count"] = len(self.weather_gdf)

        if strategy == "nearest":
            assigned = self._assign_nearest(result, self.weather_gdf, working_crs=working_crs)
        elif strategy == "idw":
            assigned = self._assign_idw(
                result,
                self.weather_gdf,
                working_crs=working_crs,
                power=idw_power,
                k=idw_k,
            )
        else:
            raise ValueError(f"Unsupported assignment strategy: {strategy}")

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

        for _, point_row in assigned_gdf.iterrows():
            point_id = point_row[id_column]
            point_slug = _slugify(point_id)

            point_dir = output_dir / point_slug
            point_dir.mkdir(parents=True, exist_ok=True)

            point_weather_gdf = self._build_point_weather_rows(
                point_row=point_row,
                point_id=point_id,
            )

            out_path = point_dir / f"{point_slug}.geojson"
            point_weather_gdf.to_file(out_path, driver=driver, index=False)
            exported_files.append(out_path)

        manifest_path = output_dir / "manifest.json"
        manifest = {
            "files_count": len(exported_files),
            "files": [str(path.relative_to(output_dir)) for path in exported_files],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.success(f"Per-point weather files exported: {len(exported_files)}")
        return exported_files

    def _assign_nearest(
        self,
        coastline_gdf: gpd.GeoDataFrame,
        weather_gdf: gpd.GeoDataFrame,
        working_crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        left = coastline_gdf.copy()
        right = weather_gdf.copy()

        metric_crs = working_crs or left.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError("Failed to determine metric CRS for nearest assignment")

        left_metric = left.to_crs(metric_crs)
        right_metric = right.to_crs(metric_crs)

        joined = gpd.sjoin_nearest(
            left_metric,
            right_metric,
            how="left",
            distance_col="weather_distance_m",
        )

        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])

        joined = joined.to_crs(left.crs)
        return joined

    def _assign_idw(
        self,
        coastline_gdf: gpd.GeoDataFrame,
        weather_gdf: gpd.GeoDataFrame,
        working_crs: str | None = None,
        power: float = 2.0,
        k: int = 4,
    ) -> gpd.GeoDataFrame:
        left = coastline_gdf.copy()
        right = weather_gdf.copy()

        metric_crs = working_crs or left.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError("Failed to determine metric CRS for IDW assignment")

        left_metric = left.to_crs(metric_crs)
        right_metric = right.to_crs(metric_crs)

        if right_metric.empty:
            raise ValueError("Weather grid is empty")

        right_xy = pd.DataFrame(
            {
                "x": right_metric.geometry.x,
                "y": right_metric.geometry.y,
            },
            index=right_metric.index,
        )

        assigned_rows: list[dict[str, Any]] = []

        for idx, row in left_metric.iterrows():
            px = row.geometry.x
            py = row.geometry.y

            distances = ((right_xy["x"] - px) ** 2 + (right_xy["y"] - py) ** 2) ** 0.5
            nearest_idx = distances.nsmallest(min(k, len(distances))).index

            nearest = right.loc[nearest_idx].copy()
            nearest_dist = distances.loc[nearest_idx]

            weights = 1.0 / nearest_dist.clip(lower=1e-9).pow(power)
            weights = weights / weights.sum()

            result_row = row.drop(labels="geometry").to_dict()
            result_row["geometry"] = coastline_gdf.loc[idx].geometry
            result_row["weather_distance_m"] = float(nearest_dist.min())
            result_row["idw_k"] = int(len(nearest))
            result_row["idw_power"] = float(power)

            for col in nearest.columns:
                if col == "geometry":
                    continue

                sample_value = nearest.iloc[0][col]

                if self._is_scalar_series(nearest[col]):
                    try:
                        result_row[col] = self._weighted_pick_or_average(nearest[col], weights)
                    except Exception:
                        result_row[col] = sample_value

            assigned_rows.append(result_row)

        assigned = gpd.GeoDataFrame(assigned_rows, geometry="geometry", crs=coastline_gdf.crs)
        return assigned

    def _build_point_weather_rows(
        self,
        point_row: pd.Series,
        point_id: Any,
    ) -> gpd.GeoDataFrame:
        geometry = point_row.geometry

        weather_payload = None
        for candidate in ("weather_timeseries", "timeseries", "weather_daily", "daily_json"):
            if candidate in point_row.index and pd.notna(point_row[candidate]):
                weather_payload = point_row[candidate]
                break

        records = self._parse_weather_payload(weather_payload)

        if not records:
            row = {
                "point_id": point_id,
                "row_no": 0,
                "date": None,
                "geometry": geometry,
            }
            return gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326")

        prepared_rows: list[dict[str, Any]] = []
        for i, record in enumerate(records):
            new_row = {"point_id": point_id, "row_no": i, "geometry": geometry}
            new_row.update(record)
            prepared_rows.append(new_row)

        return gpd.GeoDataFrame(prepared_rows, geometry="geometry", crs="EPSG:4326")

    @staticmethod
    def _parse_weather_payload(payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            return []

        if isinstance(payload, str):
            payload = payload.strip()
            if not payload:
                return []
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return []

        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        if "records" in payload and isinstance(payload["records"], list):
            return [item for item in payload["records"] if isinstance(item, dict)]

        if "daily" in payload and isinstance(payload["daily"], dict):
            daily = payload["daily"]
            time_values = daily.get("time", [])
            variable_names = [k for k in daily.keys() if k != "time"]

            records: list[dict[str, Any]] = []
            for i, dt in enumerate(time_values):
                rec = {"date": dt}
                for var_name in variable_names:
                    values = daily.get(var_name, [])
                    rec[var_name] = values[i] if i < len(values) else None
                records.append(rec)
            return records

        if "hourly" in payload and isinstance(payload["hourly"], dict):
            hourly = payload["hourly"]
            time_values = hourly.get("time", [])
            variable_names = [k for k in hourly.keys() if k != "time"]

            records = []
            for i, dt in enumerate(time_values):
                rec = {"time": dt}
                for var_name in variable_names:
                    values = hourly.get(var_name, [])
                    rec[var_name] = values[i] if i < len(values) else None
                records.append(rec)
            return records

        return []

    @staticmethod
    def _is_scalar_series(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return True
        sample = non_null.iloc[0]
        return isinstance(sample, (int, float, str, bool)) and not isinstance(sample, (dict, list, tuple))

    @staticmethod
    def _weighted_pick_or_average(series: pd.Series, weights: pd.Series) -> Any:
        non_null = series.dropna()
        if non_null.empty:
            return None

        sample = non_null.iloc[0]
        if isinstance(sample, (int, float)):
            aligned_weights = weights.loc[non_null.index]
            return float((non_null.astype(float) * aligned_weights).sum())

        return non_null.iloc[0]
