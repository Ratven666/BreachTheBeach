from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WindFetchPaths:
    main_coastline_path: str
    other_coastline_path: str | None
    points_with_normals_path: str


@dataclass(slots=True)
class WindFetchResult:
    """
    Результат трассировки одного луча
    для одной точки и одного азимута.

    ray_azimuth_deg:
        азимут луча, по которому выполнялась трассировка.

    normal_azimuth_deg:
        азимут нормали береговой линии в исходной точке.
    """

    point_id: int

    source_point_lon: float
    source_point_lat: float

    start_point_lon: float
    start_point_lat: float

    ray_azimuth_deg: float
    normal_azimuth_deg: float

    fetch_length_m: float

    hit_found: bool
    hit_lon: float | None
    hit_lat: float | None

    used_default_value: bool


@dataclass(slots=True)
class MultiDirectionFetchResult:
    """
    Результат расчёта для одной исходной точки
    и одного абсолютного азимута.

    Если азимут попадает в сухопутный сектор относительно
    нормали, трассировка не выполняется, а fetch_length_m
    принимается равным offset_m.
    """

    point_id: int
    direction_id: int

    normal_azimuth_deg: float
    azimuth_deg: float

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
