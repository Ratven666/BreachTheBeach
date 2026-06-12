from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.weather_history.domain import WeatherLayerWrapper
from src.weather_history.wheather_downloaders.open_meteo import (
    WeatherDownloadConfig,
    WeatherHistoryService,
)


def main() -> None:
    output_dir = Path("../nvrsk_calc")
    cache_dir = Path("../data/cache/open_meteo")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # normal_points_geojson_path = Path(
    #     "nvrsk_calc/nvrsk_equal_radius_200m_points_with_normals.geojson"
    # )
    points_geojson_path = Path(
        "../nvrsk_calc/nvrsk_equal_radius_200m_points.geojson"
    )

    weather_grid_geojson_path = output_dir / "weather_daily_grid_for_normal_points.geojson"
    points_with_weather_geojson_path = output_dir / "normal_points_with_weather.geojson"

    weather_start_date = "1940-01-01"
    weather_end_date = "2026-12-31"

    assignment_strategy = "idw"   # "nearest" | "idw"
    idw_power = 2.0
    idw_k = 4

    working_crs = "EPSG:32637"

    if not points_geojson_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {points_geojson_path}"
        )

    logger.info(f"Input normal points: {points_geojson_path}")
    logger.info(f"Working CRS: {working_crs}")

    weather_service = WeatherHistoryService(
        config=WeatherDownloadConfig(
            cache_dir=cache_dir,
            output_geojson_path=weather_grid_geojson_path,
            model="era5",
            grid_step=0.25,
            grid_center_offset=0.125,
            cover_points_with_cells=True,
            extra_border_cells=1,
            batch_size=20,
            timezone="GMT",
            cell_selection="nearest",
            daily_variables=(
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
            ),
        )
    )

    weather_result = weather_service.download_from_geojson(
        geojson_path=points_geojson_path,
        start_date=weather_start_date,
        end_date=weather_end_date,
        output_geojson_path=weather_grid_geojson_path,
    )

    logger.info(f"Source bbox: {weather_result['source_bbox']}")
    logger.info(f"Weather bbox: {weather_result['weather_bbox']}")
    logger.info(f"Weather grid points: {weather_result['grid_points_count']}")
    logger.success(f"Weather grid saved: {weather_result['output_geojson_path']}")

    weather_wrapper = WeatherLayerWrapper.from_file(weather_grid_geojson_path)

    assigned_gdf = weather_wrapper.assign_to_points(
        coastline_points_path=points_geojson_path,
        strategy=assignment_strategy,
        output_geojson_path=points_with_weather_geojson_path,
        output_gpkg_path=None,
        output_layer_name="normal_points_weather",
        idw_power=idw_power,
        idw_k=idw_k,
        working_crs=working_crs,
    )

    logger.info(f"Assigned strategy: {assignment_strategy}")
    logger.info(f"Assigned points count: {len(assigned_gdf)}")
    logger.success(f"Assignment GeoJSON saved: {points_with_weather_geojson_path}")
    logger.success("Weather pipeline for normal points completed successfully.")


if __name__ == "__main__":
    main()
