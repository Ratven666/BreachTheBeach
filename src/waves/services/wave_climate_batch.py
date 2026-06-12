from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.waves.services.wave_climate_service import WaveClimateService


@dataclass
class WaveClimateBatchProcessor:
    overwater_factor: float = 1.1
    breaking_coeff: float = 0.55
    rho_water: float = 1025.0
    g: float = 9.81
    bathy_radius_m: float = 20_000.0
    bathy_n_steps: int = 200
    # Дефолтные глубины при отсутствии батиметрии — пробрасываются до NearshoreWaveTransformer
    default_h_deep_m: float = 20.0
    default_h_point_m: float = 3.0

    def _point_id_col(self, gdf: gpd.GeoDataFrame) -> str:
        for candidate in ["point_id", "id", "pointid", "fid"]:
            if candidate in gdf.columns:
                return candidate
        gdf["point_id"] = range(1, len(gdf) + 1)
        return "point_id"

    @staticmethod
    def _normalize_fetch_df(df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "azimuth_deg": "direction",
            "fetch_length_m": "fetch_m",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    def run(
        self,
        points_gdf: gpd.GeoDataFrame,
        fetch_df: pd.DataFrame,
        point_weather_df: pd.DataFrame,
        normal_field: str = "normal_azimuth_deg",
        bathymetry_service: object | None = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        points = points_gdf.copy()
        point_id_col = self._point_id_col(points)

        fetch_df = self._normalize_fetch_df(fetch_df.copy())
        weather = point_weather_df.copy()

        if point_id_col not in weather.columns and "point_id" in weather.columns:
            weather = weather.rename(columns={"point_id": point_id_col})
        if point_id_col not in fetch_df.columns and "point_id" in fetch_df.columns:
            fetch_df = fetch_df.rename(columns={"point_id": point_id_col})

        daily_features: list[dict] = []
        summary_features: list[dict] = []

        point_geom = points.set_index(point_id_col).geometry.to_dict()
        point_normals = points.set_index(point_id_col)[normal_field].to_dict()

        for point_id, wdf in weather.groupby(point_id_col):
            if point_id not in point_geom or point_id not in point_normals:
                continue

            tdf = fetch_df[fetch_df[point_id_col] == point_id][
                ["direction", "fetch_m"]
            ].copy()
            if tdf.empty:
                continue

            geom = point_geom[point_id]

            svc = WaveClimateService(
                trace_df=tdf,
                wind_ts_df=wdf,
                shore_normal_deg=float(point_normals[point_id]),
                bathymetry_service=bathymetry_service,
                origin_lon=float(geom.x),
                origin_lat=float(geom.y),
                bathy_radius_m=self.bathy_radius_m,
                bathy_n_steps=self.bathy_n_steps,
                overwater_factor=self.overwater_factor,
                breaking_coeff=self.breaking_coeff,
                rho_water=self.rho_water,
                g=self.g,
                default_h_deep_m=self.default_h_deep_m,    # ← исправлено
                default_h_point_m=self.default_h_point_m,  # ← исправлено
            )
            daily = svc.calculate_daily()
            if daily.empty:
                continue

            daily[point_id_col] = point_id
            daily["shore_normal_deg"] = float(point_normals[point_id])

            for _, row in daily.iterrows():
                props = row.to_dict()
                if "date" in props and hasattr(props["date"], "isoformat"):
                    props["date"] = props["date"].isoformat()
                props["geometry"] = geom
                daily_features.append(props)

            stats = svc.cwef_stats(daily)
            stats[point_id_col] = point_id
            stats["geometry"] = geom
            summary_features.append(stats)

        daily_gdf = gpd.GeoDataFrame(daily_features, geometry="geometry", crs=points.crs)
        summary_gdf = gpd.GeoDataFrame(summary_features, geometry="geometry", crs=points.crs)
        return daily_gdf, summary_gdf

    def export(
        self,
        points_path: str | Path,
        fetch_csv_path: str | Path,
        weather_csv_path: str | Path,
        daily_output_path: str | Path,
        summary_output_path: str | Path,
        normal_field: str = "normal_azimuth_deg",
        bathymetry_service: object | None = None,
    ) -> tuple[Path, Path]:
        points_gdf = gpd.read_file(points_path)
        fetch_df = pd.read_csv(fetch_csv_path)
        weather_df = pd.read_csv(weather_csv_path)
        weather_df["date"] = pd.to_datetime(weather_df["date"])

        daily_gdf, summary_gdf = self.run(
            points_gdf,
            fetch_df,
            weather_df,
            normal_field=normal_field,
            bathymetry_service=bathymetry_service,
        )

        daily_output_path = Path(daily_output_path)
        summary_output_path = Path(summary_output_path)
        daily_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)

        daily_gdf.to_file(daily_output_path, driver="GeoJSON")
        summary_gdf.to_file(summary_output_path, driver="GeoJSON")
        return daily_output_path, summary_output_path
