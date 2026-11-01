from dataclasses import dataclass
import math

import numpy as np

from rasterio.transform import rowcol
from rasterio.windows import Window

from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class NearbyWetPixel:
    depth_m: float
    distance_m: float
    row: int
    column: int
    lon: float
    lat: float


def find_nearby_wet_pixel(
    raster,
    raster_to_lonlat,
    geometry,
    query_lon,
    query_lat,
    query_raster_x,
    query_raster_y,
    maximum_distance_m,
    window_radius,
    fill=None,
):
    """Ближайший реальный морской пиксель.

    Кандидат должен:
    - иметь конечную отрицательную elevation;
    - находиться в пределах метрического радиуса;
    - находиться на водной стороне береговой модели;
    - не находиться внутри текущего намыва;
    - соединяться с точкой запроса без пересечения
      исходного берега и текущего намыва.
    """
    if not (
        math.isfinite(maximum_distance_m)
        and maximum_distance_m > 0
    ):
        raise ValueError(
            "Радиус поиска должен быть конечным и > 0"
        )

    if type(window_radius) is not int or window_radius < 1:
        raise ValueError("window_radius должно быть целым >= 1")

    rows, columns = rowcol(
        raster.transform,
        [query_raster_x],
        [query_raster_y],
        op=np.floor,
    )

    center_row = int(np.asarray(rows).reshape(-1)[0])
    center_column = int(np.asarray(columns).reshape(-1)[0])

    row_start = max(0, center_row - window_radius)
    row_stop = min(raster.height, center_row + window_radius + 1)

    column_start = max(0, center_column - window_radius)
    column_stop = min(
        raster.width, center_column + window_radius + 1
    )

    if row_start >= row_stop or column_start >= column_stop:
        return None

    window = Window(
        col_off=column_start,
        row_off=row_start,
        width=column_stop - column_start,
        height=row_stop - row_start,
    )

    elevation = (
        raster.read(1, window=window, masked=True)
        .astype(float)
        .filled(np.nan)
    )

    wet = np.isfinite(elevation) & (elevation < 0)
    local_rows, local_columns = np.where(wet)

    if not len(local_rows):
        return None

    query_x, query_y = geometry.to_metric.transform(
        query_lon, query_lat
    )
    query_point = Point(query_x, query_y)

    has_fill = fill is not None and not fill.is_empty

    if has_fill:
        query_is_water = geometry.scenario_water(
            query_point, fill
        )
    else:
        query_is_water = geometry.original_water(query_point)

    if not query_is_water:
        return None

    candidates = []

    for local_row, local_column in zip(
        local_rows, local_columns
    ):
        row = int(row_start + local_row)
        column = int(column_start + local_column)

        raster_x, raster_y = raster.transform * (
            column + 0.5,
            row + 0.5,
        )

        lon, lat = raster_to_lonlat.transform(
            raster_x, raster_y
        )

        _, _, distance = geometry.geod.inv(
            query_lon, query_lat, lon, lat
        )
        distance = abs(float(distance))

        if (
            not math.isfinite(distance)
            or distance > maximum_distance_m
        ):
            continue

        metric_x, metric_y = geometry.to_metric.transform(
            lon, lat
        )
        candidate_point = Point(metric_x, metric_y)

        if has_fill:
            candidate_is_water = geometry.scenario_water(
                candidate_point, fill
            )
        else:
            candidate_is_water = geometry.original_water(
                candidate_point
            )

        if not candidate_is_water:
            continue

        if query_point.distance(candidate_point) > 1e-9:
            connector = LineString([
                (query_point.x, query_point.y),
                (candidate_point.x, candidate_point.y),
            ])

            if connector.intersects(
                geometry.fetch_obstacles
            ):
                continue

            if has_fill and connector.intersects(fill):
                continue

        candidates.append(
            NearbyWetPixel(
                depth_m=float(
                    -elevation[local_row, local_column]
                ),
                distance_m=distance,
                row=row,
                column=column,
                lon=float(lon),
                lat=float(lat),
            )
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: (
            item.distance_m,
            item.row,
            item.column,
        ),
    )