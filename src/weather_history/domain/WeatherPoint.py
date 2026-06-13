# src/weather_history/domain/WeatherPoint.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.weather_history.wind_rose.WindRoseBuilder import WindRose, WindRoseBuilder


# ─────────────────────────────────────────────────────────────────────────────
# WeatherTimeSeriesRow — вспомогательный тип одной записи тайм-серии
# Требуется для __init__.py и внешнего API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class WeatherTimeSeriesRow:
    """Одна строка тайм-серии погодной точки."""
    point_id: Any
    date: str | None
    wind_speed: float | None
    wind_dir: float | None
    ws_unit: str | None
    wd_unit: str | None
    geometry: Point


# Внешний кэш: (point_id, nsector) → WindRose
# Вынесен ЗА пределы датакласса, чтобы не нарушать контракт frozen=True
_WIND_ROSE_CACHE: dict[tuple[Any, int], WindRose] = {}


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

    dates: tuple[str | None, ...]
    wind_speed: tuple[float | None, ...]
    wind_dir: tuple[float | None, ...]

    ws_unit: str | None
    wd_unit: str | None
    start_date: Any
    end_date: Any

    # ── wind rose ──────────────────────────────────────────────────────────

    def build_wind_rose(self, nsector: int = 16) -> WindRose:
        key = (self.point_id, nsector)
        if key in _WIND_ROSE_CACHE:
            return _WIND_ROSE_CACHE[key]
        speeds, directions = self._valid_speed_dir_arrays()
        rose = WindRoseBuilder(nsector=nsector).build(speeds, directions)
        _WIND_ROSE_CACHE[key] = rose
        return rose

    @property
    def wind_rose(self) -> WindRose:
        return self.build_wind_rose(nsector=16)

    # ── вспомогательные ───────────────────────────────────────────────────

    def _valid_speed_dir_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        paired = [
            (s, d)
            for s, d in zip(self.wind_speed, self.wind_dir)
            if s is not None and d is not None
        ]
        if not paired:
            return np.array([], dtype=float), np.array([], dtype=float)
        speeds, dirs = zip(*paired)
        return np.array(speeds, dtype=float), np.array(dirs, dtype=float)

    def to_timeseries_rows(self) -> list[WeatherTimeSeriesRow]:
        """Возвращает список WeatherTimeSeriesRow — по одному на запись тайм-серии."""
        return [
            WeatherTimeSeriesRow(
                point_id=self.point_id,
                date=d,
                wind_speed=s,
                wind_dir=wd,
                ws_unit=self.ws_unit,
                wd_unit=self.wd_unit,
                geometry=self.geometry,
            )
            for d, s, wd in zip(self.dates, self.wind_speed, self.wind_dir)
        ]

    def to_timeseries_gdf(self, crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
        rows = [
            {
                "point_id": r.point_id,
                "date": r.date,
                "wind_speed": r.wind_speed,
                "wind_dir": r.wind_dir,
                "ws_unit": r.ws_unit,
                "wd_unit": r.wd_unit,
                "geometry": r.geometry,
            }
            for r in self.to_timeseries_rows()
        ]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    def to_summary_series(self) -> pd.Series:
        speeds, dirs = self._valid_speed_dir_arrays()
        return pd.Series({
            "point_id": self.point_id,
            "lat": self.geometry.y,
            "lon": self.geometry.x,
            "n_records": len(speeds),
            "mean_speed": float(np.mean(speeds)) if len(speeds) > 0 else None,
            "max_speed": float(np.max(speeds)) if len(speeds) > 0 else None,
            "source_grid_point_id": self.source_grid_point_id,
            "weather_distance_m": self.weather_distance_m,
        })
