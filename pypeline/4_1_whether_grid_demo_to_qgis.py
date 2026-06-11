from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, box


def _load_as_wgs84(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Input file is empty: {path}")

    if gdf.crs is None:
        raise ValueError(f"Input file has no CRS: {path}")

    return gdf.to_crs("EPSG:4326")


def _snap_down(value: float, step: float, offset: float) -> float:
    return math.floor((value - offset) / step) * step + offset


def _snap_up(value: float, step: float, offset: float) -> float:
    return math.ceil((value - offset) / step) * step + offset


def _build_grid_extent(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    grid_step: float,
    grid_center_offset: float,
    cover_points_with_cells: bool,
    extra_border_cells: int,
) -> tuple[float, float, float, float]:
    if cover_points_with_cells:
        x_start = _snap_down(minx, grid_step, grid_center_offset)
        x_end = _snap_up(maxx, grid_step, grid_center_offset)
        y_start = _snap_down(miny, grid_step, grid_center_offset)
        y_end = _snap_up(maxy, grid_step, grid_center_offset)
    else:
        x_start = _snap_up(minx, grid_step, grid_center_offset)
        x_end = _snap_down(maxx, grid_step, grid_center_offset)
        y_start = _snap_up(miny, grid_step, grid_center_offset)
        y_end = _snap_down(maxy, grid_step, grid_center_offset)

    x_start -= extra_border_cells * grid_step
    x_end += extra_border_cells * grid_step
    y_start -= extra_border_cells * grid_step
    y_end += extra_border_cells * grid_step

    return x_start, y_start, x_end, y_end


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 12))
        current += step
    return values


def build_weather_grid_lines(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    grid_step: float = 0.25,
    grid_center_offset: float = 0.125,
    cover_points_with_cells: bool = True,
    extra_border_cells: int = 1,
) -> tuple[list[dict], tuple[float, float, float, float]]:
    x_start, y_start, x_end, y_end = _build_grid_extent(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        grid_step=grid_step,
        grid_center_offset=grid_center_offset,
        cover_points_with_cells=cover_points_with_cells,
        extra_border_cells=extra_border_cells,
    )

    rows: list[dict] = []

    vertical_x = _frange(x_start, x_end, grid_step)
    horizontal_y = _frange(y_start, y_end, grid_step)

    for idx, x in enumerate(vertical_x):
        rows.append(
            {
                "layer": "weather_grid_line",
                "line_type": "vertical",
                "line_id": f"V{idx}",
                "grid_step_deg": grid_step,
                "grid_center_offset_deg": grid_center_offset,
                "coord": float(x),
                "geometry": LineString([(x, y_start), (x, y_end)]),
            }
        )

    for idx, y in enumerate(horizontal_y):
        rows.append(
            {
                "layer": "weather_grid_line",
                "line_type": "horizontal",
                "line_id": f"H{idx}",
                "grid_step_deg": grid_step,
                "grid_center_offset_deg": grid_center_offset,
                "coord": float(y),
                "geometry": LineString([(x_start, y), (x_end, y)]),
            }
        )

    return rows, (x_start, y_start, x_end, y_end)


def main() -> None:
    input_points_path = Path("../nvrsk_calc/nvrsk_equal_radius_200m_points.geojson")
    output_dir = Path("../nvrsk_calc/for_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_output_path = output_dir / "coastline_bbox.geojson"
    grid_lines_output_path = output_dir / "weather_grid_lines.geojson"
    combined_output_path = output_dir / "coastline_bbox_and_weather_grid_lines.geojson"

    grid_step = 0.25
    grid_center_offset = 0.125
    cover_points_with_cells = True
    extra_border_cells = 1

    gdf = _load_as_wgs84(input_points_path)
    minx, miny, maxx, maxy = gdf.total_bounds

    logger.info(f"Input file: {input_points_path}")
    logger.info(f"Input features: {len(gdf)}")
    logger.info(f"Coastline bbox in WGS84: {minx}, {miny}, {maxx}, {maxy}")

    bbox_geom = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame(
        [
            {
                "layer": "coastline_bbox",
                "source_file": input_points_path.name,
                "min_lon": float(minx),
                "min_lat": float(miny),
                "max_lon": float(maxx),
                "max_lat": float(maxy),
                "geometry": bbox_geom,
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    grid_rows, grid_extent = build_weather_grid_lines(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        grid_step=grid_step,
        grid_center_offset=grid_center_offset,
        cover_points_with_cells=cover_points_with_cells,
        extra_border_cells=extra_border_cells,
    )

    grid_lines_gdf = gpd.GeoDataFrame(
        grid_rows,
        geometry="geometry",
        crs="EPSG:4326",
    )

    combined_gdf = gpd.GeoDataFrame(
        list(bbox_gdf.to_dict("records")) + list(grid_lines_gdf.to_dict("records")),
        geometry="geometry",
        crs="EPSG:4326",
    )

    bbox_gdf.to_file(bbox_output_path, driver="GeoJSON")
    grid_lines_gdf.to_file(grid_lines_output_path, driver="GeoJSON")
    combined_gdf.to_file(combined_output_path, driver="GeoJSON")

    logger.info(f"Weather grid extent: {grid_extent}")
    logger.info(f"Grid line count: {len(grid_lines_gdf)}")
    logger.success(f"Saved bbox: {bbox_output_path}")
    logger.success(f"Saved weather grid lines: {grid_lines_output_path}")
    logger.success(f"Saved combined layer: {combined_output_path}")


if __name__ == "__main__":
    main()
