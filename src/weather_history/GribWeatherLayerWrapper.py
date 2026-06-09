from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cfgrib
import geopandas as gpd
import pandas as pd
import xarray as xr
from loguru import logger


@dataclass(slots=True)
class GribWeatherLayerWrapper:
    source_path: Path
    dataset: xr.Dataset | None = None
    datasets: list[xr.Dataset] | None = None
    weather_gdf: gpd.GeoDataFrame = field(
        default_factory=lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    weather_timeseries_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    preferred_payload_columns: tuple[str, ...] = (
        "node_id",
        "time",
        "valid_time",
        "u10",
        "v10",
        "latitude",
        "longitude",
        "step",
        "number",
        "surface",
    )

    @classmethod
    def from_grib(cls, path: str | Path) -> "GribWeatherLayerWrapper":
        path = Path(path)
        dataset = xr.open_dataset(path, engine="cfgrib")
        weather_timeseries_df = cls._dataset_to_timeseries_df(dataset)
        weather_gdf = cls._timeseries_to_grid_gdf(weather_timeseries_df)
        logger.info(f"Loaded GRIB dataset from {path}")
        return cls(
            source_path=path,
            dataset=dataset,
            datasets=None,
            weather_gdf=weather_gdf,
            weather_timeseries_df=weather_timeseries_df,
        )

    @classmethod
    def from_grib_datasets(cls, path: str | Path) -> "GribWeatherLayerWrapper":
        path = Path(path)
        datasets = cfgrib.open_datasets(path)

        ts_frames: list[pd.DataFrame] = []
        for ds in datasets:
            try:
                ts = cls._dataset_to_timeseries_df(ds)
                if not ts.empty:
                    ts_frames.append(ts)
            except Exception as exc:
                logger.warning(f"Skipping GRIB dataset chunk: {exc}")

        if ts_frames:
            weather_timeseries_df = pd.concat(ts_frames, ignore_index=True)
            weather_timeseries_df = weather_timeseries_df.drop_duplicates().reset_index(drop=True)
            if "node_id" in weather_timeseries_df.columns:
                weather_timeseries_df["node_id"] = weather_timeseries_df["node_id"].astype("category")
            weather_gdf = cls._timeseries_to_grid_gdf(weather_timeseries_df)
        else:
            weather_timeseries_df = pd.DataFrame()
            weather_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        logger.info(f"Loaded {len(datasets)} GRIB datasets from {path}")
        return cls(
            source_path=path,
            dataset=None,
            datasets=datasets,
            weather_gdf=weather_gdf,
            weather_timeseries_df=weather_timeseries_df,
        )

    @staticmethod
    def _find_first_existing(columns, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    @classmethod
    def _dataset_to_timeseries_df(cls, dataset: xr.Dataset) -> pd.DataFrame:
        df = dataset.to_dataframe().reset_index()

        lon_col = cls._find_first_existing(df.columns, ["longitude", "lon", "LONGITUDE"])
        lat_col = cls._find_first_existing(df.columns, ["latitude", "lat", "LATITUDE"])

        if lon_col is None or lat_col is None:
            raise ValueError("Could not detect latitude/longitude columns in GRIB dataset")

        df = df.dropna(subset=[lon_col, lat_col]).copy()

        if lon_col != "longitude":
            df = df.rename(columns={lon_col: "longitude"})
        if lat_col != "latitude":
            df = df.rename(columns={lat_col: "latitude"})

        df["node_id"] = (
            df["latitude"].round(6).astype(str)
            + "_"
            + df["longitude"].round(6).astype(str)
        )

        df["node_id"] = df["node_id"].astype("category")
        return df.reset_index(drop=True)

    @staticmethod
    def _timeseries_to_grid_gdf(timeseries_df: pd.DataFrame) -> gpd.GeoDataFrame:
        if timeseries_df.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        grid_df = (
            timeseries_df[["node_id", "latitude", "longitude"]]
            .drop_duplicates(subset=["node_id"])
            .reset_index(drop=True)
        )

        geometry = gpd.points_from_xy(grid_df["longitude"], grid_df["latitude"])
        return gpd.GeoDataFrame(grid_df, geometry=geometry, crs="EPSG:4326")

    @property
    def is_multi_dataset(self) -> bool:
        return self.datasets is not None

    def get_primary_dataset(self) -> xr.Dataset:
        if self.dataset is not None:
            return self.dataset
        if self.datasets:
            return self.datasets[0]
        raise ValueError("No GRIB dataset loaded")

    def get_datasets(self) -> list[xr.Dataset]:
        if self.datasets is not None:
            return self.datasets
        if self.dataset is not None:
            return [self.dataset]
        raise ValueError("No GRIB datasets loaded")

    def _get_timeseries_payload(self) -> pd.DataFrame:
        if self.weather_timeseries_df.empty:
            return pd.DataFrame()

        cols = [col for col in self.preferred_payload_columns if col in self.weather_timeseries_df.columns]
        payload = self.weather_timeseries_df[cols].copy()

        if "node_id" in payload.columns and payload["node_id"].dtype == "object":
            payload["node_id"] = payload["node_id"].astype("category")

        return payload

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
        aggregate_numeric: bool = True,
        max_distance: float | None = 30000.0,
    ) -> gpd.GeoDataFrame:
        if self.weather_gdf is None or self.weather_gdf.empty:
            raise ValueError("weather_gdf is empty")

        if self.weather_timeseries_df is None or self.weather_timeseries_df.empty:
            raise ValueError("weather_timeseries_df is empty")

        coastline_points_path = Path(coastline_points_path)
        coastline_gdf = gpd.read_file(coastline_points_path)
        if coastline_gdf.empty:
            raise ValueError(f"Coastline points file is empty: {coastline_points_path}")

        if coastline_gdf.crs is None:
            coastline_gdf = coastline_gdf.set_crs("EPSG:4326")

        if self.weather_gdf.crs is None:
            self.weather_gdf = self.weather_gdf.set_crs("EPSG:4326")

        target_crs = working_crs or str(coastline_gdf.crs)

        coastline_metric = coastline_gdf.to_crs(target_crs).copy()
        weather_metric = self.weather_gdf.to_crs(target_crs).copy()

        if "point_id" not in coastline_metric.columns:
            coastline_metric = coastline_metric.reset_index(drop=True)
            coastline_metric["point_id"] = coastline_metric.index.astype(str)

        strategy = strategy.lower().strip()

        if strategy == "nearest":
            assigned_metric = self._assign_nearest(
                coastline_metric=coastline_metric,
                weather_metric=weather_metric,
                max_distance=max_distance,
            )
        elif strategy == "idw":
            assigned_metric = self._assign_idw(
                coastline_metric=coastline_metric,
                weather_metric=weather_metric,
                power=idw_power,
                k=idw_k,
                aggregate_numeric=aggregate_numeric,
            )
        else:
            raise ValueError(f"Unsupported assignment strategy: {strategy}")

        assigned_wgs84 = assigned_metric.to_crs("EPSG:4326")

        if output_geojson_path is not None:
            output_geojson_path = Path(output_geojson_path)
            output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
            if len(assigned_wgs84) < 50000:
                assigned_wgs84.to_file(output_geojson_path, driver="GeoJSON")
                logger.info(f"Saved GeoJSON assignment: {output_geojson_path}")
            else:
                logger.warning(
                    f"Skip GeoJSON write for large dataset ({len(assigned_wgs84)} rows): {output_geojson_path}"
                )

        if output_gpkg_path is not None:
            output_gpkg_path = Path(output_gpkg_path)
            output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
            assigned_wgs84.to_file(output_gpkg_path, layer=output_layer_name, driver="GPKG")
            logger.info(f"Saved GPKG assignment: {output_gpkg_path}")

        return assigned_wgs84

    def _assign_nearest(
        self,
        coastline_metric: gpd.GeoDataFrame,
        weather_metric: gpd.GeoDataFrame,
        max_distance: float | None = 30000.0,
    ) -> gpd.GeoDataFrame:
        right_nodes = weather_metric[["node_id", "latitude", "longitude", "geometry"]].copy()

        node_join = gpd.sjoin_nearest(
            coastline_metric,
            right_nodes,
            how="left",
            distance_col="distance_m",
            max_distance=max_distance,
        )

        if "index_right" in node_join.columns:
            node_join = node_join.drop(columns=["index_right"])

        mapping = node_join[["point_id", "node_id", "distance_m", "geometry"]].copy()

        payload = self._get_timeseries_payload()
        if payload.empty:
            raise ValueError("Timeseries payload is empty")

        if payload["node_id"].dtype == "object":
            payload["node_id"] = payload["node_id"].astype("category")
        mapping["node_id"] = mapping["node_id"].astype(payload["node_id"].dtype)

        merged = mapping.merge(
            payload,
            on="node_id",
            how="left",
            sort=False,
            copy=False,
        )

        merged = merged.rename(columns={"geometry": "coast_geometry"})
        merged = gpd.GeoDataFrame(merged, geometry="coast_geometry", crs=coastline_metric.crs)
        merged = merged.rename_geometry("geometry")

        return merged.reset_index(drop=True)

    def _assign_idw(
        self,
        coastline_metric: gpd.GeoDataFrame,
        weather_metric: gpd.GeoDataFrame,
        power: float = 2.0,
        k: int = 4,
        aggregate_numeric: bool = True,
    ) -> gpd.GeoDataFrame:
        ts = self._get_timeseries_payload().copy()
        if ts.empty:
            raise ValueError("Timeseries payload is empty")

        time_col = "valid_time" if "valid_time" in ts.columns else ("time" if "time" in ts.columns else None)
        if time_col is None:
            raise ValueError("weather_timeseries_df has neither 'valid_time' nor 'time' column")

        numeric_cols = ts.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in {"latitude", "longitude"}]

        records: list[dict] = []

        for _, coast_row in coastline_metric.iterrows():
            distances = weather_metric.geometry.distance(coast_row.geometry)
            nearest_idx = distances.nsmallest(min(k, len(distances))).index

            neighbors = weather_metric.loc[nearest_idx].copy()
            neighbors["distance_m"] = distances.loc[nearest_idx].values
            neighbors["distance_m"] = neighbors["distance_m"].clip(lower=1e-9)
            neighbors["weight"] = 1.0 / (neighbors["distance_m"] ** power)

            ts_subset = ts[ts["node_id"].isin(neighbors["node_id"])].copy()
            if ts_subset.empty:
                row = coast_row.to_dict()
                row["geometry"] = coast_row.geometry
                records.append(row)
                continue

            ts_subset = ts_subset.merge(
                neighbors[["node_id", "distance_m", "weight"]],
                on="node_id",
                how="left",
                sort=False,
                copy=False,
            )

            for moment, group in ts_subset.groupby(time_col, dropna=False, sort=False):
                row = coast_row.to_dict()
                row[time_col] = moment

                if "time" in group.columns:
                    row["time"] = group["time"].iloc[0]
                if "valid_time" in group.columns:
                    row["valid_time"] = group["valid_time"].iloc[0]

                if aggregate_numeric:
                    for col in numeric_cols:
                        values = pd.to_numeric(group[col], errors="coerce")
                        valid = values.notna()
                        if valid.any():
                            row[col] = (
                                values[valid] * group.loc[valid, "weight"]
                            ).sum() / group.loc[valid, "weight"].sum()
                        else:
                            row[col] = None

                representative = group.sort_values("distance_m").iloc[0]
                for col in group.columns:
                    if col in row or col in {"node_id", "weight", "distance_m"}:
                        continue
                    if col in numeric_cols and aggregate_numeric:
                        continue
                    row[col] = representative[col]

                row["neighbors_used"] = int(group["node_id"].nunique())
                row["distance_m"] = float(group["distance_m"].min())
                row["geometry"] = coast_row.geometry
                records.append(row)

        return gpd.GeoDataFrame(records, geometry="geometry", crs=coastline_metric.crs)

    def export_point_files(
        self,
        assigned_gdf: gpd.GeoDataFrame,
        output_dir: str | Path,
        coast_id_column: str = "point_id",
        driver: str = "GeoJSON",
        max_files: int | None = None,
    ) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if assigned_gdf.empty:
            return []

        if coast_id_column not in assigned_gdf.columns:
            raise ValueError(f"Column '{coast_id_column}' not found in assigned_gdf")

        exported_files: list[Path] = []

        grouped = assigned_gdf.groupby(coast_id_column, sort=False)
        for idx, (coast_id, group) in enumerate(grouped):
            if max_files is not None and idx >= max_files:
                logger.warning(f"Per-point export truncated at max_files={max_files}")
                break

            safe_id = str(coast_id).replace("/", "_").replace("\\", "_").replace(" ", "_")

            if driver.lower() == "geojson":
                output_path = output_dir / f"{safe_id}.geojson"
                group.to_file(output_path, driver="GeoJSON")
            elif driver.lower() == "gpkg":
                output_path = output_dir / f"{safe_id}.gpkg"
                group.to_file(output_path, layer=safe_id[:60], driver="GPKG")
            else:
                raise ValueError(f"Unsupported driver: {driver}")

            exported_files.append(output_path)

        return exported_files

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()
        if self.datasets is not None:
            for ds in self.datasets:
                ds.close()

    def __enter__(self) -> "GribWeatherLayerWrapper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
