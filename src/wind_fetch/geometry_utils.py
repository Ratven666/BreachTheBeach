from __future__ import annotations

from math import isfinite

from pyproj import Geod
from shapely.geometry import LineString, MultiLineString, Point

WGS84_GEOD = Geod(ellps="WGS84")


def era5_from_to_ray_azimuth(era5_from_deg: float) -> float:
    """
    ERA5 / meteorological convention:
    0 = wind blowing FROM north, 90 = FROM east, etc.
    For fetch tracing we need the direction TO which the air moves.
    """
    return (float(era5_from_deg) + 180.0) % 360.0


def geodesic_forward_point(lon: float, lat: float, azimuth_deg: float, distance_m: float) -> tuple[float, float]:
    lon2, lat2, _ = WGS84_GEOD.fwd(lon, lat, azimuth_deg, distance_m)
    return float(lon2), float(lat2)


def geodesic_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, dist = WGS84_GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist)


def build_geodesic_linestring(
    lon: float,
    lat: float,
    azimuth_deg: float,
    total_length_m: float,
    step_m: float,
    max_segments: int,
) -> LineString:
    if total_length_m <= 0:
        raise ValueError("total_length_m must be > 0")
    if step_m <= 0:
        raise ValueError("step_m must be > 0")

    n_steps = max(1, min(max_segments, int(total_length_m // step_m) + 1))
    coords: list[tuple[float, float]] = [(lon, lat)]

    for i in range(1, n_steps + 1):
        dist = min(total_length_m, i * step_m)
        lon_i, lat_i = geodesic_forward_point(lon, lat, azimuth_deg, dist)
        coords.append((lon_i, lat_i))

        if dist >= total_length_m:
            break

    if coords[-1] != coords[0]:
        lon_end, lat_end = geodesic_forward_point(lon, lat, azimuth_deg, total_length_m)
        if coords[-1] != (lon_end, lat_end):
            coords.append((lon_end, lat_end))

    return LineString(coords)


def iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
        return
    if isinstance(geom, MultiLineString):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
        return
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from iter_lines(part)


def extract_points_from_intersection(geom):
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Point):
        return [geom]

    points = []
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            points.extend(extract_points_from_intersection(part))
        return points

    if hasattr(geom, "coords"):
        try:
            coords = list(geom.coords)
        except Exception:
            coords = []
        for x, y in coords:
            if isfinite(x) and isfinite(y):
                points.append(Point(x, y))
    return points
