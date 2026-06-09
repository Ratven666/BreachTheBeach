from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


@dataclass(frozen=True, slots=True)
class WeatherTimeSeriesRow:
    point_id: Any
    row_no: int
    date: str | None
    wind_speed: float | None
    wind_direction: float | None
    ws_unit: str | None
    wd_unit: str | None


@dataclass(frozen=True, slots=True)
class WeatherPoint:
    point_id: Any
    geometry: Point

    weather_strategy: str | None
    weather_distance_m: float | None

    source_grid_point_id: Any
    source_lat: float | None
    source_lon: float | None
    source_req_lat: float | None
    source_req_lon: float | None

    dates: tuple[str | None, ...] = field(default_factory=tuple)
    wind_speed: tuple[float | None, ...] = field(default_factory=tuple)
    wind_dir: tuple[float | None, ...] = field(default_factory=tuple)

    ws_unit: str | None = None
    wd_unit: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    @property
    def x(self) -> float:
        return float(self.geometry.x)

    @property
    def y(self) -> float:
        return float(self.geometry.y)

    @property
    def records_count(self) -> int:
        return len(self.dates)

    @property
    def has_weather(self) -> bool:
        return len(self.dates) > 0

    def timeseries_rows(self) -> list[WeatherTimeSeriesRow]:
        rows: list[WeatherTimeSeriesRow] = []
        n = len(self.dates)

        for i in range(n):
            rows.append(
                WeatherTimeSeriesRow(
                    point_id=self.point_id,
                    row_no=i,
                    date=self.dates[i] if i < len(self.dates) else None,
                    wind_speed=self.wind_speed[i] if i < len(self.wind_speed) else None,
                    wind_direction=self.wind_dir[i] if i < len(self.wind_dir) else None,
                    ws_unit=self.ws_unit,
                    wd_unit=self.wd_unit,
                )
            )
        return rows

    def timeseries_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "point_id": row.point_id,
                    "row_no": row.row_no,
                    "date": row.date,
                    "wind_speed": row.wind_speed,
                    "wind_direction": row.wind_direction,
                    "ws_unit": row.ws_unit,
                    "wd_unit": row.wd_unit,
                }
                for row in self.timeseries_rows()
            ]
        )

    def to_timeseries_gdf(self, crs: Any = None) -> gpd.GeoDataFrame:
        records = []

        for row in self.timeseries_rows():
            records.append(
                {
                    "point_id": row.point_id,
                    "row_no": row.row_no,
                    "date": row.date,
                    "wind_speed": row.wind_speed,
                    "wind_direction": row.wind_direction,
                    "ws_unit": row.ws_unit,
                    "wd_unit": row.wd_unit,
                    "weather_strategy": self.weather_strategy,
                    "weather_distance_m": self.weather_distance_m,
                    "source_grid_point_id": self.source_grid_point_id,
                    "source_lat": self.source_lat,
                    "source_lon": self.source_lon,
                    "source_req_lat": self.source_req_lat,
                    "source_req_lon": self.source_req_lon,
                    "geometry": self.geometry,
                }
            )

        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    def brief_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "x": self.x,
            "y": self.y,
            "records_count": self.records_count,
            "weather_strategy": self.weather_strategy,
            "weather_distance_m": self.weather_distance_m,
            "source_grid_point_id": self.source_grid_point_id,
        }
