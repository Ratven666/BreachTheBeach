# src/wind_fetch/models.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WindFetchPaths:
    main_coastline_path: str
    other_coastline_path: str
    points_with_normals_path: str


@dataclass(slots=True)
class WindFetchResult:
    """
    Результат трассировки одного луча для одной точки и одного азимута.
    ray_azimuth_deg  — азимут самого луча (тот, по которому шла трассировка).
    normal_azimuth_deg — нормаль берега в исходной точке (опорный азимут).
    """

    point_id: int

    source_point_lon: float
    source_point_lat: float

    start_point_lon: float
    start_point_lat: float

    ray_azimuth_deg: float        # был azimuth_deg — переименован
    normal_azimuth_deg: float     # новое поле

    fetch_length_m: float

    hit_found: bool
    hit_lon: float | None
    hit_lat: float | None

    used_default_value: bool


@dataclass(slots=True)
class MultiDirectionFetchResult:
    """
    Один результат = одна исходная точка + один абсолютный азимут.

    Если азимут попадает в сухопутный сектор относительно нормали,
    честная трассировка не выполняется, а fetch_length_m = offset_m.
    """

    point_id: int
    direction_id: int

    normal_azimuth_deg: float
    azimuth_deg: float            # ray azimuth (абсолютный)

    source_point_lon: float
    source_point_lat: float

    start_point_lon: float
    start_point_lat: float

    fetch_length_m: float

    hit_found: bool
    hit_lon: float | None
    hit_lat: float | None

    used_default_value: bool
    skipped_by_land_sector: bool
