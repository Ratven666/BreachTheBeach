# src/wind_fetch/geometry_utils.py
from __future__ import annotations

import math
from typing import Sequence

from pyproj import Geod

from shapely.geometry import (
    LineString,
    MultiLineString,
    GeometryCollection,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

_GEOD = Geod(ellps="WGS84")

def iter_lines(geom: BaseGeometry):
    """
    Рекурсивно извлекает все LineString-компоненты из произвольной геометрии.
    Поддерживает LineString, MultiLineString, Polygon (границы),
    MultiPolygon и GeometryCollection.
    """
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, LineString):
        yield geom

    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            yield from iter_lines(part)

    elif isinstance(geom, Polygon):
        # Внешняя граница
        yield LineString(geom.exterior.coords)
        # Внутренние границы (дыры)
        for interior in geom.interiors:
            yield LineString(interior.coords)

    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield from iter_lines(poly)

    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            yield from iter_lines(part)

def geodesic_forward_point(
    lon: float,
    lat: float,
    azimuth_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """Возвращает (lon, lat) точки, отстоящей на distance_m по азимуту azimuth_deg."""
    end_lon, end_lat, _ = _GEOD.fwd(lon, lat, azimuth_deg, distance_m)
    return float(end_lon), float(end_lat)


def normalize_azimuths(
    azimuths: Sequence[float | int],
) -> list[float]:
    """
    Приводит список азимутов к диапазону [0, 360).
    Единственная реализация — используется всеми калькуляторами.
    """
    return [float(a) % 360.0 for a in azimuths]


def normalize_angle(value: float | int) -> float:
    """Приводит одиночный угол к [0, 360)."""
    return float(value) % 360.0


def azimuth_to_dxdy(azimuth_deg: float) -> tuple[float, float]:
    """Единичный вектор направления (dx, dy) по азимуту (CW от севера)."""
    rad = math.radians(azimuth_deg)
    return math.sin(rad), math.cos(rad)


def is_in_land_sector(
    ray_azimuth_deg: float,
    normal_azimuth_deg: float,
    half_sector_deg: float,
) -> bool:
    """
    True, если ray_azimuth_deg попадает в сухопутный сектор
    ±half_sector_deg вокруг нормали, направленной в сторону суши
    (т.е. противоположной нормали к морю).
    """
    land_normal = (normal_azimuth_deg + 180.0) % 360.0
    diff = abs((ray_azimuth_deg - land_normal + 180.0) % 360.0 - 180.0)
    return diff <= half_sector_deg
