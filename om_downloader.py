from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.coastline.domain import CoastlineDataset
from src.coastline.exporters import GeoJsonPointExporter
from src.coastline.point_strategies import EqualRadiusStrategy, PointSource
from src.coastline.services import CoastlinePointExtractor
from src.weather_history.domain import WeatherLayerWrapper

from src.weather_history.wheather_downloaders.open_meteo import (
    WeatherDownloadConfig,
    WeatherHistoryService,
)


def main() -> None:
    output_dir = Path("output")
    cache_dir = Path("data/cache/open_meteo")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    main_coastline_path = "data/main_coastline.geojson"
    other_lines_path = "data/other_lines.geojson"

    radius_points_geojson_path = output_dir / "novoross_main_radius_points.geojson"
    weather_grid_geojson_path = output_dir / "weather_daily_grid.geojson"

    coastline_weather_geojson_path = output_dir / "coastline_points_with_weather.geojson"
    coastline_weather_gpkg_path = output_dir / "coastline_points_with_weather.gpkg"
    coastline_weather_per_point_dir = output_dir / "coastline_weather_points_rows"

    weather_start_date = "1940-01-01"
    weather_end_date = "2026-12-31"

    assignment_strategy = "idw"   # "nearest" | "idw"
    idw_power = 2.0
    idw_k = 4

    dataset = CoastlineDataset.from_geojson(
        main_path=main_coastline_path,
        other_path=other_lines_path,
        name="novoross_coastline",
    )
    dataset.print_summary()

    main_gdf = dataset.main_gdf
    if main_gdf.crs is None:
        raise ValueError("dataset.main_gdf has no CRS")

    working_crs = main_gdf.estimate_utm_crs()
    if working_crs is None:
        raise ValueError("Failed to estimate working metric CRS from main coastline")

    working_crs_str = str(working_crs)
    logger.info(f"Working metric CRS: {working_crs_str}")

    extractor = CoastlinePointExtractor()

    radius_points = extractor.extract(
        dataset=dataset,
        strategy=EqualRadiusStrategy(
            radius_step_m=500.0,
            source=PointSource.MAIN_ONLY,
            include_origin=True,
            include_endpoint=True,
            working_crs=working_crs_str,
        ),
        name="novoross_main_radius_points",
    )

    radius_points.print_summary()

    radius_points.export(
        strategy=GeoJsonPointExporter(),
        output_path=str(radius_points_geojson_path),
    )
    logger.success(f"Coastline radius points saved: {radius_points_geojson_path}")

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
        geojson_path=radius_points_geojson_path,
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
        coastline_points_path=radius_points_geojson_path,
        strategy=assignment_strategy,
        output_geojson_path=coastline_weather_geojson_path,
        output_gpkg_path=coastline_weather_gpkg_path,
        output_layer_name="coastline_weather_points",
        idw_power=idw_power,
        idw_k=idw_k,
        working_crs=working_crs_str,
    )

    logger.info(f"Assigned strategy: {assignment_strategy}")
    logger.info(f"Assigned coastline points count: {len(assigned_gdf)}")
    logger.success(f"Assignment GeoJSON saved: {coastline_weather_geojson_path}")
    logger.success(f"Assignment GPKG saved: {coastline_weather_gpkg_path}")

    exported_files = weather_wrapper.export_point_files(
        assigned_gdf=assigned_gdf,
        output_dir=coastline_weather_per_point_dir,
        coast_id_column="point_id",  # если такого поля нет, метод сам сделает fallback
        driver="GeoJSON",
    )

    logger.info(f"Per-point files count: {len(exported_files)}")
    logger.success(f"Per-point directory saved: {coastline_weather_per_point_dir}")
    logger.success("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
