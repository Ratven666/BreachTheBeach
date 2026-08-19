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
    wind_speed_max: float | None
    wind_speed_mean: float | None
    wind_gust_max: float | None
    wind_dir: float | None
    ws_max_unit: str | None
    ws_mean_unit: str | None
    wg_max_unit: str | None
    wd_unit: str | None
    geometry: Point


# Внешний кэш: (point_id, nsector, speed_field) → WindRose
# Вынесен ЗА пределы датакласса, чтобы не нарушать контракт frozen=True
_WIND_ROSE_CACHE: dict[tuple[Any, int, str], WindRose] = {}


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
    wind_speed_max: tuple[float | None, ...]
    wind_speed_mean: tuple[float | None, ...]
    wind_gust_max: tuple[float | None, ...]
    wind_dir: tuple[float | None, ...]

    ws_max_unit: str | None
    ws_mean_unit: str | None
    wg_max_unit: str | None
    wd_unit: str | None
    start_date: Any
    end_date: Any

    # ── wind rose ──────────────────────────────────────────────────────────

    def build_wind_rose(self, nsector: int = 16, speed_field: str = "mean") -> WindRose:
        key = (self.point_id, nsector, speed_field)
        if key in _WIND_ROSE_CACHE:
            return _WIND_ROSE_CACHE[key]
        speeds, directions = self._valid_speed_dir_arrays(speed_field=speed_field)
        rose = WindRoseBuilder(nsector=nsector).build(speeds, directions)
        _WIND_ROSE_CACHE[key] = rose
        return rose

    @property
    def wind_rose(self) -> WindRose:
        """Роза ветров по средней скорости — стандартная климатологическая картина."""
        return self.build_wind_rose(nsector=16, speed_field="mean")

    @property
    def wind_rose_max(self) -> WindRose:
        """Роза ветров по суточному максимуму средней скорости."""
        return self.build_wind_rose(nsector=16, speed_field="max")

    @property
    def wind_rose_gust(self) -> WindRose:
        """Роза ветров по порывам — для оценки экстремальной ветровой нагрузки."""
        return self.build_wind_rose(nsector=16, speed_field="gust")

    # ── вспомогательные ───────────────────────────────────────────────────

    def _speed_series(self, speed_field: str) -> tuple[float | None, ...]:
        if speed_field == "mean":
            return self.wind_speed_mean
        if speed_field == "max":
            return self.wind_speed_max
        if speed_field == "gust":
            return self.wind_gust_max
        raise ValueError(f"Unsupported speed_field: {speed_field}")

    def _valid_speed_dir_arrays(self, speed_field: str = "mean") -> tuple[np.ndarray, np.ndarray]:
        speed_series = self._speed_series(speed_field)
        paired = [
            (s, d)
            for s, d in zip(speed_series, self.wind_dir)
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
                wind_speed_max=ws_max,
                wind_speed_mean=ws_mean,
                wind_gust_max=wg_max,
                wind_dir=wd,
                ws_max_unit=self.ws_max_unit,
                ws_mean_unit=self.ws_mean_unit,
                wg_max_unit=self.wg_max_unit,
                wd_unit=self.wd_unit,
                geometry=self.geometry,
            )
            for d, ws_max, ws_mean, wg_max, wd in zip(
                self.dates,
                self.wind_speed_max,
                self.wind_speed_mean,
                self.wind_gust_max,
                self.wind_dir,
            )
        ]

    def to_timeseries_gdf(self, crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
        rows = [
            {
                "point_id": r.point_id,
                "date": r.date,
                "wind_speed_max": r.wind_speed_max,
                "wind_speed_mean": r.wind_speed_mean,
                "wind_gust_max": r.wind_gust_max,
                "wind_dir": r.wind_dir,
                "ws_max_unit": r.ws_max_unit,
                "ws_mean_unit": r.ws_mean_unit,
                "wg_max_unit": r.wg_max_unit,
                "wd_unit": r.wd_unit,
                "geometry": r.geometry,
            }
            for r in self.to_timeseries_rows()
        ]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    def to_summary_series(self) -> pd.Series:
        mean_speeds, _ = self._valid_speed_dir_arrays(speed_field="mean")
        max_speeds, _ = self._valid_speed_dir_arrays(speed_field="max")
        gusts, _ = self._valid_speed_dir_arrays(speed_field="gust")

        return pd.Series({
            "point_id": self.point_id,
            "lat": self.geometry.y,
            "lon": self.geometry.x,
            "n_records": len(mean_speeds),
            "mean_speed": float(np.mean(mean_speeds)) if len(mean_speeds) > 0 else None,
            "max_speed": float(np.max(max_speeds)) if len(max_speeds) > 0 else None,
            "max_gust": float(np.max(gusts)) if len(gusts) > 0 else None,
            "source_grid_point_id": self.source_grid_point_id,
            "weather_distance_m": self.weather_distance_m,
        })
