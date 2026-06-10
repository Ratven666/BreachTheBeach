from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.weather_history.wind_rose import (
    MatplotlibWindRosePlotter,
    PlotlyWindRosePlotter,
    WindRose,
    WindRoseBuilder,
)


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

    _wind_rose_cache: WindRose | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_wind_rose_cache", None)

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
        return len(self.dates) > 0 and len(self.wind_speed) > 0 and len(self.wind_dir) > 0

    @property
    def wind_rose(self) -> WindRose:
        cached = self._wind_rose_cache
        if cached is not None:
            return cached

        rose = self.build_wind_rose(
            nsector=16,
            bins=None,
            calm_limit=None,
            title=f"Wind rose for point {self.point_id}",
        )
        object.__setattr__(self, "_wind_rose_cache", rose)
        return rose

    def build_wind_rose(
        self,
        nsector: int = 16,
        bins: int | list[float] | tuple[float, ...] | None = None,
        calm_limit: float | None = None,
        title: str | None = None,
    ) -> WindRose:
        if len(self.wind_speed) != len(self.wind_dir):
            raise ValueError(
                f"wind_speed and wind_dir length mismatch for point {self.point_id}: "
                f"{len(self.wind_speed)} != {len(self.wind_dir)}"
            )

        builder = WindRoseBuilder(
            nsector=nsector,
            bins=bins,
            calm_limit=calm_limit,
        )
        return builder.build(
            speed=self.wind_speed,
            direction=self.wind_dir,
            ws_unit=self.ws_unit,
            title=title or f"Wind rose for point {self.point_id}",
        )

    def plot_wind_rose_matplotlib(
        self,
        output_path: str | None = None,
        nsector: int = 16,
        bins: int | list[float] | tuple[float, ...] | None = None,
        calm_limit: float | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (8, 8),
    ):
        rose = self.build_wind_rose(
            nsector=nsector,
            bins=bins,
            calm_limit=calm_limit,
        )
        plotter = MatplotlibWindRosePlotter()

        if output_path is None:
            return plotter.plot_bar(
                wind_rose=rose,
                figsize=figsize,
                cmap=cmap,
            )

        return plotter.save_bar(
            wind_rose=rose,
            output_path=output_path,
            figsize=figsize,
            cmap=cmap,
        )

    def plot_wind_rose_plotly(
        self,
        output_path: str | None = None,
        nsector: int = 16,
        bins: int | list[float] | tuple[float, ...] | None = None,
        calm_limit: float | None = None,
    ):
        rose = self.build_wind_rose(
            nsector=nsector,
            bins=bins,
            calm_limit=calm_limit,
        )
        plotter = PlotlyWindRosePlotter()

        if output_path is None:
            return plotter.build_barpolar(rose)

        return plotter.save_html(rose, output_path=output_path)

    def timeseries_rows(self) -> list[WeatherTimeSeriesRow]:
        rows: list[WeatherTimeSeriesRow] = []
        n = min(len(self.dates), len(self.wind_speed), len(self.wind_dir))

        for i in range(n):
            rows.append(
                WeatherTimeSeriesRow(
                    point_id=self.point_id,
                    row_no=i,
                    date=self.dates[i],
                    wind_speed=self.wind_speed[i],
                    wind_direction=self.wind_dir[i],
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
        records: list[dict[str, Any]] = []

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
                    "start_date": self.start_date,
                    "end_date": self.end_date,
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
            "has_weather": self.has_weather,
        }